import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import torch
from torch import nn
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

## Set paths to data and output directories
DATA_SOURCE = '/nfs/data/1/rrazakami/work/dune-cvn/dataset_info'
# DATA_ROOT = 'data_cvn'
# DATA_ROOT = DATA_SOURCE
DATA_ROOT ='/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd'
os.makedirs(DATA_ROOT, exist_ok=True)
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fix_random_seeds(seed=53):
    '''
        Fix random seeds for reproducibility of the results.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device():
    '''
        Check if cuda is available and return the device accordingly.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'Device count : {torch.cuda.device_count()}')
    return device

## class for DINOv2
class ModelWithIntermediateLayers(nn.Module):
    '''
        Get the n last blocks from the dinov2 model. This is used to extract the features for the linear classifier. 
        Features:
            - patch tokens
            - class token
    '''
    def __init__(self, feature_model, n_last_blocks):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks

    def forward(self, images):
        # with torch.inference_mode():
        features = self.feature_model.get_intermediate_layers(images, self.n_last_blocks, return_class_token=True)
        return features
    
    