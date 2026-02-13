import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

import os, time, yaml
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import Subset

from dinov2.data import DataAugmentationDINO, SamplerType, make_data_loader, make_dataset
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
from dinov2.eval.visualize_attention import get_attn
from PIL import Image

def fix_random_seeds(seed=53):
    '''
        Fix the random seeds for reproducibility. This function sets the random seed for PyTorch, CUDA, and NumPy
          to ensure that the results are consistent across different runs of the code.
        Args:
            seed: the random seed to be set for PyTorch, CUDA, and NumPy
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def prepare_dataset(params_dict, img_size):
    '''
        Prepare the dataset for training and evaluation. This function creates the dataset using the specified parameters, 
        splits it into training and validation sets, and creates data loaders for both sets.
        Args:
            params_dict: dictionary containing the parameters for dataset preparation, including the data directory, batch size, 
            number of workers, image size, patch size, and data augmentation parameters
        Returns:
            train_loader: data loader for the training set
            val_loader: data loader for the validation set
    '''
    DATA_ROOT = params_dict['data_root']
    OUTPUT_DIR = params_dict['output_dir']
    BATCH_SIZE = params_dict['batch_size']
    NUM_WORKERS = params_dict['num_workers']
    ## Try to create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data_transform = DataAugmentationDINO(
        global_crops_scale=1.0,
        global_crops_size=img_size,
        local_crops_size=img_size,
    )

    datasets_full = make_dataset(dataset_str=f"CVN:root={DATA_ROOT}:extra=plane=Z,mono=mono3", transform=data_transform,
                                target_transform=None, classification_type=params_dict['classification_type']) ## Pass classification_type to make_dataset to ensure the dataset is created with the correct target variable
    class_names = datasets_full.classes ## Get class names from the dataset
    print(f'Class names: {class_names}')
    total_size = len(datasets_full)
    print(f'Total number of samples in the dataset: {total_size}')
    print(f'Selecting a subset of size: {params_dict["N_SAMPLES"]} for training, evaluation, and testing')

    # Randomly select N indices
    np.random.seed(42)
    selected_indices = np.random.choice(total_size, size=params_dict['N_SAMPLES'], replace=False)

    # Get labels for the selected samples (needed for stratification)
    selected_labels = []
    for idx in selected_indices:
        _, label = datasets_full[idx]
        selected_labels.append(label if isinstance(label, int) else label.item())
    selected_labels = np.array(selected_labels)

    # First split: train (70%) vs temp (30%)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        selected_indices,
        selected_labels,
        test_size=0.3,
        stratify=selected_labels,
        random_state=42
    )

    # Second split: val (15%) and test (15%) from temp
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    # Create subsets
    datasets_train = Subset(datasets_full, train_indices)
    datasets_val = Subset(datasets_full, val_indices)
    datasets_test = Subset(datasets_full, test_indices)
    print(f'Number of training samples: {len(datasets_train)}')
    print(f'Number of validation samples: {len(datasets_val)}')
    print(f'Number of test samples: {len(datasets_test)}')

    def nclasses_in_dataset(dataset, class_names):
        class_set = {class_names[i]: 0 for i in range(len(class_names))}
        for _, label in dataset:
            class_name = dataset.dataset.classes[label]
            class_set[class_name] += 1
        return class_set

    print(f'Classes in training set: {nclasses_in_dataset(datasets_train, class_names)}')
    print(f'Classes in validation set: {nclasses_in_dataset(datasets_val, class_names)}')
    print(f'Classes in test set: {nclasses_in_dataset(datasets_test, class_names)}')

    dataset_splitting = {
        'train': nclasses_in_dataset(datasets_train, class_names),
        'val': nclasses_in_dataset(datasets_val, class_names),
        'test': nclasses_in_dataset(datasets_test, class_names)
    }
    dataloaders = {
        'train': make_data_loader(
            dataset=datasets_train,
            batch_size=BATCH_SIZE,
            sampler_type=SamplerType.EPOCH,
            num_workers=NUM_WORKERS,
            shuffle=True
        ),
        'val': make_data_loader(
            dataset=datasets_val,
            batch_size=BATCH_SIZE,
            sampler_type=SamplerType.EPOCH,
            num_workers=NUM_WORKERS,
            shuffle=True
        ),
    }

    return dataloaders, class_names, datasets_test, dataset_splitting


class LinearClassifierCVN(nn.Module):
    '''
        This class implements a linear classifier for the DINOv2 backbone. It takes the output from the last n blocks of the DINOv2 model, 
        concatenates the class tokens, and optionally appends the average of the patch tokens from the last block.
        The resulting tensor is then fed into a linear layer to produce the final classification output.
    '''
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
    
    def create_linear_input(self, x_tokens_list, use_nblocks, use_avgpool):
        '''
            Create the input for the linear classifier by concatenating the class tokens from the last n blocks and optionally appending the average of the patch tokens from the last block.
            Args:
                x_tokens_list: list of tuples (patch_tokens, class_token) from each block
                use_nblocks: number of blocks to use for the linear classifier
                use_avgpool: whether to append the average of the patch tokens from the last block
            Returns:
                output: tensor of shape (batch_size, out_dim) to be fed into the linear classifier
        '''
        intermediate_output = x_tokens_list[-use_nblocks:]  # list of tensors from the last n blocks
        output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)  # concatenate class tokens
        if use_avgpool:
            output = torch.cat(
                (
                    output,
                    torch.mean(intermediate_output[-1][0], dim=1), # patch tokens
                ),
                dim=-1,
            )
            output = output.reshape(output.shape[0], -1)
        return output.float()

    def forward(self, input):
        if self.use_dinov2:
            output = self.create_linear_input(input, self.use_nblocks, self.use_avgpool) # Get the cls token and append the average of the patch tokens to it
            return self.linear(output)


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

def load_config_pretraining(config_path):
    '''
        Load the configuration file from the training phase, which contains information about the DINOv2 model 
        architecture used during training. This information is crucial for loading the correct variant of the DINOv2 model in the evaluation phase.
        Args:
            config_path: path to the configuration file from the training phase
        Returns:
            config_pretraining: dictionary containing the configuration for the DINOv2 model architecture used during training
    '''
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
    return config

def load_dinov2_backbone(config_dict, config_pretraining, device='cuda:0'):
    '''
        Load the DINOv2 backbone based on the configuration dictionary. The function supports loading the small, base, and large variants of the DINOv2 model.
        Args:
            config_dict: dictionary containing the configuration for the DINOv2 backbone
            config_pretraining: dictionary containing the configuration for the DINOv2 model architecture used during training
        Returns:
            model: the loaded DINOv2 backbone model
    '''
    IMG_SIZE = config_dict['img_size']
    PATCH_SIZE = config_dict['patch_size']
    backbone_arch = config_pretraining['student']['arch']
    backbone_model = None
    if backbone_arch == 'vit_small':
        backbone_model = vit_small(patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    elif backbone_arch == 'vit_base':
        backbone_model = vit_base(patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    elif backbone_arch == 'vit_large':
        backbone_model = vit_large(patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone_arch}")
    print(f'===Backbone architecture: {backbone_arch}')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_model.to(device)
    for p in backbone_model.parameters():
        p.requires_grad = False
    backbone_model.eval()

    pth_model = torch.load(config_dict['model_path'])
    backbone_model.load_state_dict(pth_model, strict=False)
    print(f'Backbone model : {backbone_model}')

    feature_model = ModelWithIntermediateLayers(backbone_model, n_last_blocks=config_dict['use_nblocks'])
    embed_dim = backbone_model.embed_dim * (config_dict['use_nblocks'] + int(config_dict['use_avgpool']))
    feature_model = feature_model.to(device)
    return feature_model, embed_dim


def train_loop(dataloader, feature_model, linear_classifier, loss_fn, optimizer, BATCH_SIZE, device):
    '''
        Train the linear classifier for one epoch. This function iterates over the training data, 
        extracts features using the DINOv2 backbone, feeds the features into the linear classifier, 
        computes the loss, and updates the weights of the linear classifier using backpropagation.
        Args:
            dataloader: data loader for the training set
            feature_model: the DINOv2 backbone model used for feature extraction
            linear_classifier: the linear classifier model to be trained
            loss_fn: the loss function used for training
            optimizer: the optimizer used for updating the weights of the linear classifier
            BATCH_SIZE: the batch size used for training
            device: the device (CPU or GPU) on which the training is performed
        Returns:
            epoch_acc: the accuracy of the linear classifier on the training set for the current epoch
            epoch_loss: the average loss of the linear classifier on the training set for the current epoch
    '''
    linear_classifier.train()
    size = len(dataloader.dataset)
    print(f'Dataset size: {size}')
    total_samples = 0
    # num_batches = len(dataloader[0]['global_crops'])
    num_batches = BATCH_SIZE
    running_loss = 0.0
    running_corrects = 0
    for batch, (X,y) in enumerate(dataloader):
        print(f'Batch {batch+1}/{num_batches}', end='\r')
        X = X.to(device)
        y = y.to(device)
        features = feature_model(X)
        pred = linear_classifier(features)
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

def val_loop(dataloader, feature_model, linear_classifier, loss_fn, device):
    '''
        Validate the linear classifier on the validation set. This function iterates over the validation data, 
        extracts features using the DINOv2 backbone, feeds the features into the linear classifier, computes the loss, 
        and calculates the accuracy of the linear classifier on the validation set.
        Args:
            dataloader: data loader for the validation set
            feature_model: the DINOv2 backbone model used for feature extraction
            linear_classifier: the linear classifier model to be evaluated
            loss_fn: the loss function used for evaluation
            device: the device (CPU or GPU) on which the evaluation is performed
        Returns:
            val_acc: the accuracy of the linear classifier on the validation set
            val_loss: the average loss of the linear classifier on the validation set
    '''
    feature_model.eval()
    linear_classifier.eval()
    total_samples = 0
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            features = feature_model(X)
            pred = linear_classifier(features)
            val_loss += loss_fn(pred, y).item()
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += y.size(0)
    val_loss /= total_samples
    val_acc /= total_samples
    return val_acc, val_loss

def plot_training_curves(train_data, OUTPUT_DIR):
    '''
        Plot the training curves for accuracy, loss, and learning rate over the epochs. 
        This function takes the training data collected during the training process and creates plots for the training and validation accuracy, 
        training and validation loss, and learning rate. The plots are saved to the specified output directory.
        Args:
            train_data: a list of dictionaries containing the training and validation accuracy, loss, and learning rate for each epoch
            OUTPUT_DIR: the directory where the training curves will be saved
    '''
    print(f'Plotting training curves completed and saved to {OUTPUT_DIR}/training_curves.png')
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
    # plt.show()
    plt.close()

def train(dataloaders, feature_model, linear_classifier, OUTPUT_DIR, EPOCHS=100, BATCH_SIZE=32, device='cuda'):
    '''
        Train the linear classifier using the training data and validate it using the validation data. This function orchestrates the entire training process, 
        including setting up the loss function, optimizer, and learning rate scheduler, as well as saving the best model based on validation accuracy and loss.
        Args:
            dataloaders: a dictionary containing the data loaders for the training and validation sets
            feature_model: the DINOv2 backbone model used for feature extraction
            linear_classifier: the linear classifier model to be trained
            OUTPUT_DIR: the directory where the best model and training curves will be saved
            EPOCHS: the number of epochs to train the linear classifier
            BATCH_SIZE: the batch size for training
            device: the device to run the training on ('cuda' or 'cpu')
        Returns:
            feature_model: the DINOv2 backbone model used for feature extraction (unchanged during training)
            linear_classifier: the trained linear classifier model
    '''
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        # lr=cfg['optim']['base_lr'],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=0)

    best_acc = 0.0
    best_acc_loss = np.inf
    train_data = []
    for t in range(EPOCHS): # epochs
        print(f'Epoch {t+1}\n-------------------------------')

        train_acc, train_loss = train_loop(dataloaders['train'], feature_model, linear_classifier, loss_fn, optimizer, device=device, BATCH_SIZE=BATCH_SIZE)
        train_data.append({
            'phase': 'train',
            'epoch': t,
            'lr': optimizer.param_groups[0]['lr'],
            'accuracy': train_acc,
            'loss': train_loss
        })
        scheduler.step()
        print(f'Train: \n  Train_acc = {train_acc:.4f}, Train_loss = {train_loss:.4f}')
        val_acc, val_loss = val_loop(dataloaders['val'], feature_model, linear_classifier, loss_fn, device=device)
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
    print('\n\n')
    plot_training_curves(train_data=train_data, OUTPUT_DIR=OUTPUT_DIR)
    return feature_model, linear_classifier

def visualize_model(feature_model, linear_classifier, images, device, class_names, original_img=None, rows=2, cols=4,
                     true_classes=None, batch_no=None, backbone_model=None, PATCH_SIZE=16, IMG_SIZE=224):
    '''
        Visualize the predictions of the linear classifier on a batch of images. This function takes a batch of images, 
        extracts features using the DINOv2 backbone, feeds the features into the linear classifier to get the predicted classes, 
        and visualizes the predictions along with the true classes. If a backbone model is provided, it also visualizes the attention maps for the input images.
        Args:
            feature_model: the DINOv2 backbone model used for feature extraction
            linear_classifier: the linear classifier model used for prediction
            images: a batch of input images to be visualized
            device: the device (CPU or GPU) on which the visualization is performed
            class_names: a list of class names corresponding to the class indices
            original_img: the original images before any transformations (optional, used for attention visualization)
            rows: the number of rows in the visualization grid
            cols: the number of columns in the visualization grid
            true_classes: the true class labels for the input images (optional, used for displaying true class names in the visualization)
            batch_no: the batch number (optional, used for naming the output files)
            backbone_model: the DINOv2 backbone model used for feature extraction (optional, used for attention visualization)
             PATCH_SIZE: the patch size used for the DINOv2 model (used for attention visualization)
             IMG_SIZE: the image size used for the DINOv2 model (used for attention visualization)
        Returns:
            figname: the name of the saved visualization file
            predicted_class: the predicted class name for the input images
            true_class: the true class name for the input images
    '''
    was_training = linear_classifier.training
    linear_classifier.eval()
    with torch.no_grad():
        # img = img[np.newaxis, ...]
        imgs = images[np.newaxis, ...].to(device)
        features = feature_model(imgs)
        outputs = linear_classifier(features)
        outputs = nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        print(f'Predicted classes: {preds}')
        print(f'True classes: {true_classes}')
        print(f'Outputs: {outputs}')
        titles = [f'{class_names[preds[0]]}: {outputs[0, preds[0]].squeeze().item():.3f}; true : {class_names[true_classes]}']
    print(f'imgs size : {imgs.size()}')
    # Try to create output directory if it doesn't exist
    output_dir = 'output/predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    figname = f'{output_dir}/predictions{class_names[preds[0]]}_true{class_names[true_classes]}_batch{batch_no}.png' if batch_no is not None else f'{output_dir}/predictions{class_names[preds[0]]}_true{class_names[true_classes]}_batch.png'
    if original_img is not None:
        imgs = original_img
    if batch_no%2==0:
        # Print test sample every 100 events
        if backbone_model is not None:
            print(f'images shape : {images.shape}')
            print(f'type(images): {type(images)}')
            # img = np.array(images)
            print(f'img shape : {images.shape}')
            print(f'type(img) : {type(images)}')
            print('----Getting into get_attn function----')
            img = images
            attention = get_attn(model=backbone_model, event=img, patch_size=PATCH_SIZE, image_size=IMG_SIZE, sumoverheads=True)
            fig = plt.figure(figsize=(10,10))
            print(f'image shape : {img.shape}')
            i = 0
            ax = plt.subplot(rows, cols, i+1)
            plt.imsave(fname=figname.replace('.png', '_image.png'), arr=img.cpu().numpy().transpose(1,2,0)[:, :, 0], format='png')
            plt.imsave(fname=figname.replace('.png', '_attention.png'), arr=attention, format='png')

            modified_img = Image.open(figname.replace('.png', '_image.png')).convert("RGBA")
            attention_img = Image.open(figname.replace('.png', '_attention.png')).convert("L").resize(modified_img.size)
            modified_img.paste(attention_img, (0,0), attention_img)

            ax.set_title(titles[i])
            ax.axis('off')
            fig.tight_layout()
            if figname is not None:
                # plt.savefig(figname)
                # plt.close()
                modified_img.save(figname)
        else:
            if titles is None:
                titles = ['image ' + str(i) for i in range(imgs.size(0))]
            if (cols!=1) & (rows!=1):
                fig = plt.figure(figsize=(cols * 3, rows * 3))
            else:
                fig = plt.figure(figsize=(10,10))

            img = imgs[0].cpu().numpy().transpose((1,2,0))
            print(f'image shape : {img.shape}')
            i = 0
            ax = plt.subplot(rows, cols, i+1)
            ax.imshow(img[:, :, 0], cmap='viridis') ## just cmap='gray' doesn't change this to gray
            ax.set_title(titles[i])
            ax.axis('off')
            fig.tight_layout()
            if figname is not None:
                plt.savefig(figname)
                plt.close()

    linear_classifier.train(mode=was_training)
    return figname, class_names[preds[0]], class_names[true_classes]

def test(feature_model, linear_classifier, datasets_test, class_names, PATCH_SIZE, IMG_SIZE, OUTPUT_DIR, backbone_model=None, device='cuda'):
    '''
        Test the linear classifier on the test set and visualize the predictions. This function iterates over the test data, extracts features using the DINOv2 backbone, 
        feeds the features into the linear classifier to get the predicted classes, and visualizes the predictions along with the true classes. 
        It also computes the overall accuracy on the test set and saves a confusion matrix.
        Args:
            feature_model: the DINOv2 backbone model used for feature extraction
            linear_classifier: the linear classifier model used for prediction
            datasets_test: the test dataset to be evaluated
            class_names: a list of class names corresponding to the class indices
            PATCH_SIZE: the patch size used for the DINOv2 model (used for attention visualization)
            IMG_SIZE: the image size used for the DINOv2 model (used for attention visualization)
            OUTPUT_DIR: the directory where the test set predictions and confusion matrix will be saved
            backbone_model: the DINOv2 backbone model used for feature extraction (optional, used for attention visualization)
            device: the device (CPU or GPU) on which the evaluation is performed
        Returns:
            None (the function saves the predictions and confusion matrix to the specified output directory)
    '''
    print('Starting test set predictions and visualization...')
    predictions_dict = {
        'figname': [],
        'predicted_class': [],
        'true_class': []
    }
    for no, (img, label) in enumerate(datasets_test):
        figname, predicted_class, true_class = visualize_model(feature_model=feature_model, linear_classifier=linear_classifier, images=img, device=device, 
                                                               class_names=class_names, original_img=None, true_classes=label, rows=1, cols=1, batch_no=no, backbone_model=backbone_model,
                                                                 PATCH_SIZE=PATCH_SIZE, IMG_SIZE=IMG_SIZE)
        predictions_dict['figname'].append(figname)
        predictions_dict['predicted_class'].append(predicted_class)
        predictions_dict['true_class'].append(true_class)

    predictions_df = pd.DataFrame(predictions_dict)
    accuracy = (predictions_df['predicted_class'] == predictions_df['true_class']).sum() / len(predictions_df)
    print(f'Test set accuracy: {accuracy:.4f}')
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'test_set_predictions.csv'), index=False)

    cm = confusion_matrix(predictions_df['true_class'], predictions_df['predicted_class'], labels=class_names)

    f, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    config_dict = {
        'epochs': 50,
        'img_size': 224,
        'patch_size': 16,
        'use_nblocks': 1,
        'use_avgpool': True,
        'backbone_name': 'dinov2',
        # 'model_path': '/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/model_final.rank_0.pth',
        'model_path': '/nfs/data/1/rrazakami/work/OUTPUT_DINO/training_dinov2/output_Feb12_2026/model_final.rank_0.pth'
    }
    # PATH_TO_CONFIG = '/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/config.yaml'
    PATH_TO_CONFIG = '/nfs/data/1/rrazakami/work/OUTPUT_DINO/training_dinov2/output_Feb12_2026/config.yaml'

    # Data params dict
    params_dict = {
        'data_root': '/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd',
        'classification_type': 'flavor_2',
        'output_dir': 'output',
        'batch_size': 32,
        'num_workers': 0,
        'N_SAMPLES': 50000
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataloaders, class_names, datasets_test, dataseet_splitting = prepare_dataset(params_dict=params_dict, img_size=config_dict['img_size'])

    ## dump dataset_splitting into json file located at params_dict['output_dir']
    import json
    with open(os.path.join(params_dict['output_dir'], 'dataset_splitting.json'), 'w') as f:
        json.dump(dataseet_splitting, f, indent=4)
        
    config_pretraining = load_config_pretraining(PATH_TO_CONFIG)

    feature_model, embed_dim = load_dinov2_backbone(config_dict=config_dict, config_pretraining=config_pretraining, device=device)

    linear_classifier = LinearClassifierCVN(out_dim=embed_dim, config=config_dict, num_classes=len(class_names))
    linear_classifier.to(device)

    feature_model, linear_classifier = train(dataloaders=dataloaders, feature_model=feature_model, linear_classifier=linear_classifier, 
                                             OUTPUT_DIR=params_dict['output_dir'], EPOCHS=config_dict['epochs'], BATCH_SIZE=params_dict['batch_size'], device=device)

    test(feature_model=feature_model, linear_classifier=linear_classifier, datasets_test=datasets_test, class_names=class_names, 
         PATCH_SIZE=config_dict['patch_size'], IMG_SIZE=config_dict['img_size'], OUTPUT_DIR=params_dict['output_dir'], backbone_model=feature_model.feature_model, device=device)
