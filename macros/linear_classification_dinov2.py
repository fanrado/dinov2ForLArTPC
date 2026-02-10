#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from pathlib import Path
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from tqdm import tqdm
# #### Data loader: copy from cvn.py
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.models.feature_extraction import create_feature_extractor
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')

def fix_random_seeds(seed=53):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
fix_random_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
print(f'Device count : {torch.cuda.device_count()}')

# ### Preparing the file structure
DATA_SOURCE = '/nfs/data/1/rrazakami/work/dune-cvn/dataset_info'
DATA_ROOT = 'data_cvn'
os.makedirs(DATA_ROOT, exist_ok=True)
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_dir_structure():
    def get_unique_filename(flavor='nutau'):
        # return unique list of prefixes in the flavor directory
        flavor_path = os.path.join(DATA_SOURCE, flavor)
        list_files = os.listdir(flavor_path)
        prefixes = [f.split('.')[0] for f in list_files if f.startswith('event')]
        unique_prefixes = list(set(prefixes))
        return unique_prefixes
    ## The particular dataset is balanced, so each flavor has the name number of events. This could be different for other datasets.
    N = len(get_unique_filename('nutau'))
    # N = 20
    print(f'Total number of events per flavor: {N}')

    # separate into train, val, test : 70%, 15%, 15%
    train_split = int(0.7 * N)
    val_split = int(0.15 * N)
    test_split = N - train_split - val_split
    # Each event is stored in a gz file with its corresponding .info file. We need to copy both files.
    ## Create train, val, and test directories
    train_path = os.path.join(DATA_ROOT, 'train')
    val_path = os.path.join(DATA_ROOT, 'val')
    test_path = os.path.join(DATA_ROOT, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    # For each flavor, create subdirectories
    # flavors = os.listdir(DATA_SOURCE)
    flavors = ['nue', 'numu', 'nc']
    for flavor in flavors:
        os.makedirs(os.path.join(train_path, flavor), exist_ok=True)
        os.makedirs(os.path.join(val_path, flavor), exist_ok=True)
        os.makedirs(os.path.join(test_path, flavor), exist_ok=True)
    # Now, copy files into respective directories
    for flavor in flavors:
        list_files = get_unique_filename(flavor=flavor)
        flavor_path = os.path.join(DATA_SOURCE, flavor)
        for i, file in enumerate(list_files):
            src_file_gz = os.path.join(flavor_path, file + '.gz')
            src_file_info = os.path.join(flavor_path, file + '.info')
            if i < train_split:
                dest_dir = os.path.join(train_path, flavor)
                # dest_dir = os.path.join(train_path)
            elif i < train_split + val_split:
                dest_dir = os.path.join(val_path, flavor)
                # dest_dir = os.path.join(val_path)
            else:
                dest_dir = os.path.join(test_path, flavor)
                # dest_dir = os.path.join(test_path)
            shutil.copy(src_file_gz, dest_dir)
            shutil.copy(src_file_info, dest_dir)
            ##
            ## rename files
            # gz_file = src_file_gz.split('/')[-1]
            # info_file = src_file_info.split('/')[-1]
            # new_gz_name = f'{gz_file.split(".")[0]}_{flavor}.gz'
            # new_info_name = f'{info_file.split(".")[0]}_{flavor}.info'
            # src_file_gz = os.path.join(dest_dir, gz_file)
            # src_file_info = os.path.join(dest_dir, info_file)
            # dest_file_gz = os.path.join(dest_dir, new_gz_name)
            # dest_file_info = os.path.join(dest_dir, new_info_name)
            # shutil.move(src_file_gz, dest_file_gz)
            # shutil.move(src_file_info, dest_file_info)

# create_dir_structure()

# ### Creating models for feature extraction
## class for DINOv2
class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks

    def forward(self, images):
        # with torch.inference_mode():
        features = self.feature_model.get_intermediate_layers(images, self.n_last_blocks, return_class_token=True)
        return features


# ### Configuring the model
BACKBONE = 'dinov2'
model_configurator = {
    'dinov2': {
        'backbone_name': 'dinov2',
        # 'arch' : 'large', # Rado's training
        'arch': 'base', # Nitish's training
        'use_nblocks' : 1,
        'use_avgpool': True,
    }
}
# IMG_SIZE = 500 #168
# IMG_SIZE = 16*30
IMG_SIZE = 224
BATCH_SIZE = 32 #8
NUM_WORKERS = 0
# PATCH_SIZE = 25 #56
PATCH_SIZE = 16

import yaml
with open('/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/config.yaml','r') as f:
    config = yaml.safe_load(f)

model_config = model_configurator[BACKBONE]
if BACKBONE=='dinov2':
    backbone_archs = {
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large
    }
    # backbone_arch = backbone_archs[model_config['arch']]
    backbone_arch = backbone_archs[config['student']['arch']]
    backbone_name = f'dinov2_{backbone_arch}'
    print(f'Using backbone: {backbone_name}')
    # load model
    # backbone_model = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE) ## <<== This is wrong. We need to use the same arch as used in training
    backbone_model = backbone_arch(patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    backbone_model.to(device)
    for p in backbone_model.parameters():
        p.requires_grad = False
    backbone_model.eval()
    # pth_model = torch.load('../out_cvn_memlite_batchpergpu_16/model_final.rank_0.pth')
    pth_model = torch.load('/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/model_final.rank_0.pth')
    backbone_model.load_state_dict(pth_model, strict=False)
    feature_model = ModelWithIntermediateLayers(backbone_model, n_last_blocks=model_config['use_nblocks'])
    embed_dim = backbone_model.embed_dim * (model_config['use_nblocks'] + int(model_config['use_avgpool']))

feature_model = feature_model.to(device)

# ### Creating a linear classifier
def create_linear_input(x_tokens_list, use_nblocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_nblocks:]  # list of tensors from the last n blocks
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)  # concatenate class tokens
    print(f'patch tokens shape: {intermediate_output[-1][0].shape}')
    print(f'output shape before avgpool: {output.shape}')
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1), # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
        print(f'output shape after avgpool: {output.shape}')
    return output.float()

class LinearClassifier(nn.Module):
    def __init__(self, out_dim, config, num_classes=4):
        super().__init__()
        self.out_dim = out_dim
        if config['backbone_name'] == 'dinov2':
            self.use_dinov2= True
            self.use_nblocks = config['use_nblocks']
            self.use_avgpool = config['use_avgpool']
        self.num_classes = num_classes
        self.linear = nn.Linear(self.out_dim, self.num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, input):
        if self.use_dinov2:
            output = create_linear_input(input, self.use_nblocks, self.use_avgpool)
            return self.linear(output)

linear_classifier = LinearClassifier(embed_dim, model_config, num_classes=3) # nu and nubar in the same class. No nutau
linear_classifier = linear_classifier.to(device)

# ### Defining datasets and dataloaders
from dinov2.data import DataAugmentationDINO, SamplerType, make_data_loader, make_dataset
from dinov2.data import MaskingGenerator, collate_data_and_cast
from functools import partial
import yaml

with open('../dinov2/configs/ssl_default_config.yaml') as f:
    cfg = yaml.safe_load(f)
# cfg = yaml.safe_load(open('../dinov2/configs/ssl_default_config.yaml'))
cfg['train']['batch_size_per_gpu'] = BATCH_SIZE
cfg['crops']['global_crops_size'] = IMG_SIZE
cfg['student']['patch_size'] = PATCH_SIZE

img_size = cfg["crops"]["global_crops_size"]
patch_size = cfg["student"]["patch_size"]

n_tokens = (img_size // patch_size) ** 2  # +1 for CLS token
mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

data_transform = DataAugmentationDINO(
    global_crops_scale=cfg["crops"]["global_crops_scale"],
    global_crops_size=cfg["crops"]["global_crops_size"],
    local_crops_size=cfg["crops"]["local_crops_size"],
)

# collate_fn = partial(
#     collate_data_and_cast,
#     mask_ratio_tupe=cfg["ibot"]["mask_ratio_min_max"],
#     mask_probability=cfg["ibot"]["mask_sample_probability"],
#     n_tokens=n_tokens,
#     mask_generator=mask_generator,
#     dtype=torch.half,
# )

kwargs = {}
datasets = {x: make_dataset(dataset_str=f"CVN:root={DATA_ROOT}/{x}:extra=plane=Z,mono=mono3", transform=data_transform,
                            target_transform=None) for x in ['train', 'val']}
dataloaders = {x: make_data_loader(
    dataset=datasets[x],
    batch_size=cfg["train"]["batch_size_per_gpu"],
    num_workers=cfg["train"]["num_workers"],
    shuffle=True,
    sampler_type=SamplerType.EPOCH
) for x in ['train', 'val']}

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes
# class_names = datasets['train'].pdgs

print(f"Data loaded with {dataset_sizes['train']} training samples and {dataset_sizes['val']} validation samples.")
print(f'Tes')

datasets['test'] = make_dataset(dataset_str=f"CVN:root={DATA_ROOT}/test:extra=plane=Z,mono=mono3", transform=data_transform,
                            target_transform=None)
dataloaders['test'] = make_data_loader(
    dataset=datasets['test'],
    batch_size=cfg["train"]["batch_size_per_gpu"],
    num_workers=cfg["train"]["num_workers"],
    shuffle=True,
)
dataset_sizes['test'] = len(datasets['test'])
print(f'Data loaded with {dataset_sizes["test"]} test samples.')

# ### Visualizing some examples
def show_batch(imgs, titles=None, rows=2, cols=4, figname=None):
    if titles is None:
        titles = ['image ' + str(i) for i in range(imgs.size(0))]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if (cols!=1) & (rows!=1):
        fig = plt.figure(figsize=(cols * 3, rows * 3))
    else:
        fig = plt.figure(figsize=(10,10))
    # for i in range(imgs.size(0)):
    # print(f'imgs.size() : {imgs.size()}')
    # print(f'imgs.size(0) : {imgs.size(0)}')
    # img = imgs[i].cpu().numpy().transpose((1, 2, 0))
    img = imgs.cpu().numpy().transpose((1,2,0))
    print(f'image shape : {img.shape}')
    # img = img * std + mean  # unnormalize
    img = np.clip(img, 0, 1)
    i = 0
    ax = plt.subplot(rows, cols, i+1)
    ax.imshow(img) ## just cmap='gray' doesn't change this to gray
    ax.set_title(titles[i])
    ax.axis('off')
    fig.tight_layout()
    if figname is not None:
        plt.savefig(figname)
        plt.close()

inputs, classes, _ = next(iter(dataloaders['train']))

nu_PDGs = {
    12: r'$\nu_e$',
    -12: r'$\bar{\nu}_e$',
    14: r'$\nu_{\mu}$',
    -14: r'$\bar{\nu}_{\mu}$',
    16: r'$\nu_{\tau}$',
    -16: r'$\bar{\nu}_{\tau}$',
}

# ### Train and validation loops
def train_loop(dataloader, feature_model, linear_classifier, loss_fn, optimizer):
    linear_classifier.train()
    size = len(dataloader.dataset)
    print(f'Dataset size: {size}')
    total_samples = 0
    # num_batches = len(dataloader[0]['global_crops'])
    num_batches = BATCH_SIZE
    running_loss = 0.0
    running_corrects = 0
    for batch, (X,y, img_original) in enumerate(dataloader):
        print(f'Batch {batch+1}/{num_batches}', end='\r')
        X = X.to(device)
        y = y.to(device)
        features = feature_model(X)
        # f = features[0][0].cpu().numpy()
        # features = feature_model.forward(X)
        pred = linear_classifier(features)
        # print(f'pred shape : {pred.shape}, values : {pred}')
        # print(f'y shape : {y.shape}, values : {y}')
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Statistics
        running_loss += loss.item()
        running_corrects += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_samples += y.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    print(f'len(running_corrects) : {running_corrects}, total_samples : {total_samples}')
    return epoch_acc, epoch_loss

# @torch.inference_mode()
def val_loop(dataloader, feature_model, linear_classifier, loss_fn):
    feature_model.eval()
    linear_classifier.eval()
    size = len(dataloader.dataset)
    total_samples = 0
    num_batches = BATCH_SIZE
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for X, y,_ in dataloader:
            X = X.to(device)
            y = y.to(device)
            features = feature_model(X)
            # features = feature_model.forward_backward(X)
            pred = linear_classifier(features)
            val_loss += loss_fn(pred, y).item()
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += y.size(0)
    val_loss /= total_samples
    val_acc /= total_samples
    return val_acc, val_loss


# ### Loss function, optimizer, scheduler
EPOCHS = 200
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    linear_classifier.parameters(),
    # lr=cfg['optim']['base_lr'],
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=0)

best_acc = 0.0
best_acc_loss = np.inf
train_data = []
for t in range(EPOCHS): # epochs
    print(f'Epoch {t+1}\n-------------------------------')

    train_acc, train_loss = train_loop(dataloaders['train'], feature_model, linear_classifier, loss_fn, optimizer)
    train_data.append({
        'phase': 'train',
        'epoch': t,
        'lr': optimizer.param_groups[0]['lr'],
        'accuracy': train_acc,
        'loss': train_loss
    })
    scheduler.step()
    print(f'Train: \n  Train_acc = {train_acc:.4f}, Train_loss = {train_loss:.4f}')
    val_acc, val_loss = val_loop(dataloaders['val'], feature_model, linear_classifier, loss_fn)
    print(f'Validation:\n    val_acc = {val_acc}, val_loss = {val_loss}')
    train_data.append({
        'phase': 'val',
        'epoch': t,
        'lr': optimizer.param_groups[0]['lr'],
        'accuracy': val_acc,
        'loss': val_loss
    })
    if (val_acc == best_acc and val_loss < best_acc_loss) or (val_acc > best_acc):
        best_acc, best_acc_loss = val_acc, val_loss
        save_dict = {
            'epoch': t+1,
            'state_dict': linear_classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'best_acc_loss': best_acc_loss
        }
        torch.save(save_dict, os.path.join(OUTPUT_DIR, 'linear_classifier_best.pth'))
    print('\n')
print('Training completed.')

# ### Model information
import pandas as pd
train_df = pd.DataFrame(train_data)
train_df.to_csv(os.path.join(OUTPUT_DIR, 'training_log.csv'), index=False)
train_phase = train_df[train_df['phase']=='train']
val_phase = train_df[train_df['phase']=='val']
y_axis = 'accuracy'
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
line_acc_train, = ax1.plot(train_phase['epoch'], train_phase['accuracy'], 'g-', label='train acc')
line_acc_val, = ax1.plot(val_phase['epoch'], val_phase['accuracy'], 'r-', label='val acc')
line_loss_train, = ax1.plot(train_phase['epoch'], train_phase['loss'], 'g--', label='train loss')
line_loss_val, = ax1.plot(val_phase['epoch'], val_phase['loss'], 'r--', label='val loss')
ax1.set_xlabel('Epoch number')
ax1.legend(loc='center right')
ax1.set_title('Accuracy and cross entropy')
ax1.grid(visible=True)
line_lr = ax2.plot(train_phase['epoch'], train_phase['lr'], 'ob', label='learning rate')
ax2.set_xlabel('Epoch number')
ax2.set_title('Learning rate')
ax2.grid(visible=True)
fig.suptitle('Model info')
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
plt.show()


# ### Visualization of predictions
print(class_names)

def visualize_model(feature_model, linear_classifier, images,original_img=None, rows=2, cols=4, true_classes=None, batch_no=None):
    was_training = linear_classifier.training
    linear_classifier.eval()
    with torch.no_grad():
        imgs = images.to(device)
        # print(f'Input image shape: {imgs.shape}')
        features = feature_model(imgs)
        outputs = linear_classifier(features)
        outputs = nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        print(f'Predicted classes: {preds}')
        print(f'True classes: {true_classes}')
        print(f'Outputs: {outputs}')
        # titles = [f'{class_names[preds[i]]}: {outputs[i, preds[i]].squeeze().item():.3f}; true : {class_names[true_classes[i]]}' for i in range(imgs.size(0))]
        titles = [f'{class_names[preds[0]]}: {outputs[0, preds[0]].squeeze().item():.3f}; true : {class_names[true_classes]}']
    # figname = f'output/predictions/predictions_batch_{batch_no}.png' if batch_no is not None else 'output/predictions/predictions_batch.png'
    figname = f'output/predictions/predictions{class_names[preds[0]]}_true{class_names[true_classes]}_batch{batch_no}.png' if batch_no is not None else f'output/predictions/predictions{class_names[preds[0]]}_true{class_names[true_classes]}_batch.png'
    if original_img is not None:
        imgs = original_img
    show_batch(imgs, titles=titles, rows=rows, cols=cols, figname=figname)

    linear_classifier.train(mode=was_training)
    return figname, class_names[preds[0]], class_names[true_classes]

# i = 0
# for batch_no, (imgs, true_classes) in enumerate(dataloaders['test']):
#     # if i==2:
#         # break
#     # imgs, true_classes = next(iter(dataloaders['test']))
#     visualize_model(feature_model=feature_model, linear_classifier=linear_classifier, images=imgs, true_classes=true_classes, rows=4, cols=8, batch_no=batch_no)
#     i += 1

predictions_dict = {
    'figname': [],
    'predicted_class': [],
    'true_class': []
}
for no, (img, label, original_img) in enumerate(datasets['test']):
    img = img[np.newaxis, ...]
    figname, predicted_class, true_class = visualize_model(feature_model=feature_model, linear_classifier=linear_classifier, images=img, original_img=original_img, true_classes=label, rows=1, cols=1, batch_no=no)
    predictions_dict['figname'].append(figname)
    predictions_dict['predicted_class'].append(predicted_class)
    predictions_dict['true_class'].append(true_class)

predictions_df = pd.DataFrame(predictions_dict)
accuracy = (predictions_df['predicted_class'] == predictions_df['true_class']).sum() / len(predictions_df)
print(f'Test set accuracy: {accuracy:.4f}')
predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'test_set_predictions.csv'), index=False)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predictions_df['true_class'], predictions_df['predicted_class'], labels=class_names)

import seaborn as sns
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.show()
