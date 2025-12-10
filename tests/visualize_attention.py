# # Copyright (c) Facebook, Inc. and its affiliates.
# # 
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# # 
# #     http://www.apache.org/licenses/LICENSE-2.0
# # 
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # Note the original is here: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoConfig
sys.path.append('../')
from dinov2.models.vision_transformer import vit_small, vit_large

if __name__ == '__main__':
    # image_size = (952, 952)
    image_size = (480, 480)
    # image_size = (224, 224)
    output_dir = 'attn/'
    patch_size = 14

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = vit_large(
            patch_size=14,
            # img_size=224,
            init_values=1.0,
            #ffn_layer="mlp",
            block_chunks=0
    )
    # model = vit_large(patch_size=14)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    # url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    # state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    # model.load_state_dict(state_dict, strict=True)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    print('Model loaded from transformers', model)
    # sys.exit()
    # for p in model.parameters():
    #     p.requires_grad = False
    
    # model.eval()
    # model.to(device)
    print('HERE')
    img = Image.open('cow-beach.jpg')
    print(f'image size: {img.size}')
    img0 = img.convert('RGB')
    transform = pth_transforms.Compose([
        # pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img0)
    print(f'Initial image shape : {img.shape}')

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    print(f'cropped size: {(w, h)}')

    img = img[:, :w, :h].unsqueeze(0)
    print(f'final image size: {img.shape}')
    # sys.exit()
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    # w_featmap = 48
    # h_featmap = 32
    # sys.exit()
    print(f'Image size : {img.shape}')

    # attentions = model.get_last_selfattention(img.to(device))
    attentions = model.get_last_self_attention(img.to(device))
    print(f'Raw attention map shape: {attentions.shape}')
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # for every patch
    print(f'attention shape : {attentions.shape}, {nh}')
    
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # attentions = attentions[0, :, 0:].reshape(nh, -1)
    print(attentions.shape)
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1)) 
    # attentions[:, 283] = 0 

    ##
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".jpg")
        plt.imsave(fname=fname, arr=attentions[j], format='jpg')
        print(f"{fname} saved.")


