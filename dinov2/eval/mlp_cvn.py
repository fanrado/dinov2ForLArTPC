from linear_cvn import prepare_dataset
from linear_cvn import train, test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, torch, yaml
import torch.nn as nn

class MLP_Patching(nn.Module):
    def __init__(self, patch_size=16, n_classes=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.linear = nn.LazyLinear(self.n_classes) # lazy linear layer, we will initialize it in the forward pass when we know the input dimension
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
    
    def img2patch2vec(self, x):
        '''
            img2patch2vec takes an image tensor of shape (B, C, H, W) and converts it into a patch vector of shape (B, num_patches*patch_size*patch_size*C), where num_patches = (H//patch_size)*(W//patch_size). This is done by first using the unfold function to extract non-overlapping patches from the image, and then reshaping and permuting the dimensions to get the desired output shape.
        '''
        # x shape : (B, C, H, W)
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # (B, C, H//patch_size, W//patch_size, patch_size, patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size) # (B, C, num_patches, patch_size, patch_size)
        x = x.permute(0, 2, 1, 3, 4) # (B, num_patches, C, patch_size, patch_size)
        x = x.contiguous().view(B, -1, self.patch_size * self.patch_size * C) # (B, num_patches, patch_size*patch_size*C)
        ## concatenate the patch vectors into a single vector, becoming (B, num_patches*patch_size*patch_size*C), so that we can feed it to the MLP
        x = x.view(B, -1) # (B, num_patches*patch_size*patch_size*C)
        return x
    
    def forward(self, x):
        '''
            forward takes an image tensor of shape (B, C, H, W) and returns the output of the linear layer after feeding the patch vector obtained from img2patch2vec. The output shape is (B, n_classes).
        '''
        # x shape : (B, C, H, W)
        # convert x into (B, num_patches*patch_size*patch_size*C)
        x = self.img2patch2vec(x)
        # feed x to a linear layer 
        # return self.linear(x)
        return self.linear(x)


if __name__ =="__main__":
    config_dict = {
        'epochs': 50,
        'img_size': 224,
        'patch_size': 16,
        'use_nblocks': 1,
        'use_avgpool': False, #True,
        'backbone_name': 'dinov2',
        'model_path': '/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/model_final.rank_0.pth',
        # 'model_path': '/nfs/data/1/rrazakami/work/OUTPUT_DINO/training_dinov2/output_Feb12_2026/model_final.rank_0.pth'
    }
    PATH_TO_CONFIG = '/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/config.yaml'
    # PATH_TO_CONFIG = '/nfs/data/1/rrazakami/work/OUTPUT_DINO/training_dinov2/output_Feb12_2026/config.yaml'

    # Data params dict
    params_dict = {
        'data_root': '/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd',
        'classification_type': 'flavor_2',
        'output_dir': 'output/mlp',
        'batch_size': 32,
        'num_workers': 0,
        'N_SAMPLES': 10000 #50000
    }
    try:
        os.mkdir(params_dict['output_dir'])
    except FileExistsError:
        pass
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataloaders, class_names, datasets_test, dataseet_splitting = prepare_dataset(params_dict=params_dict, img_size=config_dict['img_size'])

    MLP = MLP_Patching(patch_size=config_dict['patch_size'], n_classes=len(class_names))
    MLP = MLP.to(device)

    _, linear_classifier = train(dataloaders=dataloaders, linear_classifier=MLP, OUTPUT_DIR=params_dict['output_dir'], EPOCHS=config_dict['epochs'], BATCH_SIZE=params_dict['batch_size'], device=device)

    test(feature_model=None, linear_classifier=linear_classifier, datasets_test=datasets_test, class_names=class_names, PATCH_SIZE=config_dict['patch_size'], IMG_SIZE=config_dict['img_size'], OUTPUT_DIR=params_dict['output_dir'], device=device)