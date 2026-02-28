from linear_cvn import prepare_dataset
from linear_cvn import train, test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, torch, yaml
import torch.nn as nn

class MLP_Patching(nn.Module):
    def __init__(self, patch_size=16, n_classes=4, dropout_rate=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        ## only linear mlp
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
        x = self.dropout(x) # apply dropout to the patch vector before feeding it to the linear layer, this is a common regularization technique to prevent overfitting. We can experiment with different dropout rates to see how it affects the performance of the model.
        # feed x to a linear layer 
        return self.linear(x)


class MLP_Tunable(nn.Module):
    def __init__(self, patch_size=16, n_classes=4, hidden_layers=[512, 256], 
                 activation='relu', dropout_rate=0.5, use_batch_norm=False, use_layer_norm=False):
        """
        Tunable MLP with configurable hyperparameters.
        
        Args:
            patch_size (int): Size of image patches
            n_classes (int): Number of output classes
            hidden_layers (list): List of hidden layer sizes. Empty list means no hidden layers (linear only)
            activation (str): Activation function - 'relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'
            dropout_rate (float): Dropout rate to apply after each layer
            use_batch_norm (bool): Whether to use batch normalization after each hidden layer
            use_layer_norm (bool): Whether to use layer normalization after each hidden layer
        """
        super().__init__()
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        # Build the network
        self.network = self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self):
        """Returns the activation function based on the activation_name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(self.activation_name.lower(), nn.ReLU())
    
    def _build_network(self):
        """Builds the MLP network based on hyperparameters"""
        layers = []
        
        # If no hidden layers, just use a single linear layer
        if len(self.hidden_layers) == 0:
            layers.append(nn.LazyLinear(self.n_classes))
            if self.dropout_rate > 0:
                layers.insert(0, nn.Dropout(self.dropout_rate))
        else:
            # Input dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            # First hidden layer (lazy to infer input size)
            layers.append(nn.LazyLinear(self.hidden_layers[0]))
            
            # Add normalization if specified
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_layers[0]))
            elif self.use_layer_norm:
                layers.append(nn.LayerNorm(self.hidden_layers[0]))
            
            # Add activation
            layers.append(self._get_activation())
            
            # Add subsequent hidden layers
            for i in range(1, len(self.hidden_layers)):
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                
                layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
                
                # Add normalization if specified
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
                elif self.use_layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_layers[i]))
                
                # Add activation
                layers.append(self._get_activation())
            
            # Output layer
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.Linear(self.hidden_layers[-1], self.n_classes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights for all linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
            forward takes an image tensor of shape (B, C, H, W) and returns the output of the network after feeding the patch vector obtained from img2patch2vec. The output shape is (B, n_classes).
        '''
        # x shape : (B, C, H, W)
        # convert x into (B, num_patches*patch_size*patch_size*C)
        x = self.img2patch2vec(x)
        # feed x through the network
        return self.network(x)
    
    def tune_hyperparameters(self, hidden_layers=None, activation=None, dropout_rate=None, 
                            use_batch_norm=None, use_layer_norm=None):
        """
        Tune hyperparameters and rebuild the network.
        
        Args:
            hidden_layers (list, optional): New list of hidden layer sizes
            activation (str, optional): New activation function
            dropout_rate (float, optional): New dropout rate
            use_batch_norm (bool, optional): Whether to use batch normalization
            use_layer_norm (bool, optional): Whether to use layer normalization
        
        Returns:
            self: Returns the model instance for method chaining
        """
        # Update hyperparameters if provided
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
        if activation is not None:
            self.activation_name = activation
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate
        if use_batch_norm is not None:
            self.use_batch_norm = use_batch_norm
        if use_layer_norm is not None:
            self.use_layer_norm = use_layer_norm
        
        # Rebuild the network with new hyperparameters
        self.network = self._build_network()
        
        # Re-initialize weights
        self._initialize_weights()
        
        return self
    
    def get_config(self):
        """Returns the current hyperparameter configuration"""
        return {
            'patch_size': self.patch_size,
            'n_classes': self.n_classes,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm
        }


if __name__ =="__main__":
    config_dict = {
        'epochs': 100,
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
        'N_SAMPLES': 100000 # use 100k samples for training and testing
    }
    try:
        os.mkdir(params_dict['output_dir'])
    except FileExistsError:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataloaders, class_names, datasets_test, dataseet_splitting = prepare_dataset(params_dict=params_dict, img_size=config_dict['img_size'])
    import json
    with open(os.path.join(params_dict['output_dir'], 'dataset_splitting.json'), 'w') as f:
        json.dump(dataseet_splitting, f, indent=4)
    print('Dataset prepared and splitting saved to output directory...')

    MLP = MLP_Patching(patch_size=config_dict['patch_size'], n_classes=len(class_names))
    MLP = MLP.to(device)
    print('MLP model initialized and moved to device...')

    _, linear_classifier = train(dataloaders=dataloaders, linear_classifier=MLP, OUTPUT_DIR=params_dict['output_dir'], EPOCHS=config_dict['epochs'], BATCH_SIZE=params_dict['batch_size'], device=device)
    print('MLP model trained...')
    print('Testing MLP model on test set...')
    test(feature_model=None, linear_classifier=linear_classifier, datasets_test=datasets_test, class_names=class_names, PATCH_SIZE=config_dict['patch_size'], IMG_SIZE=config_dict['img_size'], OUTPUT_DIR=params_dict['output_dir'], device=device)