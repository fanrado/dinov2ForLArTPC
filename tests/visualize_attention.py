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
import zlib
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
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoConfig
sys.path.append('../')
from dinov2.models.vision_transformer import vit_small, vit_large, vit_giant2

from fvcore.common.checkpoint import Checkpointer

def apply_mask(image, mask, color, alpha=0.5):
    new_image = image.copy()
    # if image.shape[0] == 3:
        
    # print('Applying mask to the image')
    for c in range(3):
        print(f'Image shape : {new_image.shape}')
        print(f'Mask shape : {mask.shape}')
        print(f'Color : {color[c]}')
        print(f'alpha : {alpha}')
        new_image[:, :, c] = new_image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return new_image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = np.asarray(image).astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def _read_array(gz_path, swap_axes=False, plane='Z'):
        with open(gz_path, "rb") as f:
            arr = np.frombuffer(bytearray(zlib.decompress(f.read())), dtype=np.uint8).reshape(3, 500, 500)
        if swap_axes:
            # swap wire/time -> (3, W, T) -> (3, T, W)
            arr = arr.transpose(0, 2, 1)
        # choose one plane
        idx_map = {"U": 0, "V": 1, "Z": 2, "0": 0, "1": 1, "2": 2}
        idx = idx_map[plane]
        plane = arr[idx]                                      # (H,W)
        ## Not used----- commenting out
        # if mono == "mono1":
        #     return Image.fromarray(plane, mode="L")            # true 1-channel
        ## -----------------------
        # default: replicate into 3 channels (Option A)
        img = np.repeat(plane[..., None], 3, axis=2)           # (H,W,3)
        return Image.fromarray(img, mode="RGB")
        
        return arr  # (3, H, W)
### Read the events from cvn dataset and save as png files
# path_to_gz = '/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd/prodgenie_dunevd_1x8x6_nue/cvn_gaushit/72787986_732/'
# for f in os.listdir(path_to_gz):
#     if f.endswith('.gz'):
#         path_to_file = os.path.join(path_to_gz, f)
#         event = _read_array(gz_path=path_to_file)
#         plt.imsave(fname='events_cvn/'+f.replace('.gz', '.png'), arr=np.moveaxis(event, 0, -1), format='png')
# sys.exit()
path_to_gz = '/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd/prodgenie_dunevd_1x8x6_nue/cvn_gaushit/72787986_732/event_r933_s1_e31742_h1695836876.gz'
event = _read_array(gz_path=path_to_gz)
if __name__ == '__main__':
    # image_size = (518, 518)
    image_size = (500, 500)
    # image_size = (480, 480)
    # image_size = (224, 224)
    output_dir = 'attn/'
    patch_size = 16

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dist.init_process_group(backend='gloo', init_method='env://', master_addr='localhost', master_port='12355', rank=0, world_size=1)
    # model = vit_giant2(
    #         patch_size=14,
    #         img_size=518,
    #         # init_values=1.0,
    #         #ffn_layer="mlp",
    #         # block_chunks=1
    # )
    # model = vit_large()
    #         patch_size=14,
    #         # img_size=518,
    #         # init_values=1.0,
    #         #ffn_layer="mlp",
    #         # block_chunks=1
    # )
    model = vit_large(patch_size=16, img_size=image_size[0])
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    # url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    # state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    # model.load_state_dict(state_dict, strict=True)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # dinov2_vitl14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
    # model.load_state_dict(dinov2_vitl14_lc.state_dict(), strict=False)
    ##
    ## Load model from pth file
    # pth_model = torch.load('dinov2_vitg14_pretrain.pth')
    ### ---
    ## Load a model from a training checkpoint. Training from scratch using cvn dataset
    pth_model = torch.load('../out_cvn_memlite_batchpergpu_16/model_final.rank_0.pth')
    model.load_state_dict(pth_model, strict=False)
    ## -----------------
    # pth_model = torch.load('/nfs/data/1/nitish/dino_output/small_run1_basemask/model_final.rank_0.pth',)
    # #                        map_location='cpu', weights_only=True)
    # print(pth_model)

    # pth_model = torch.load('model_final.rank_0.pth')
    print('\n')
    # print('Model loaded from pth file', pth_model['pos_embed'].shape)
    # print('Model pos_embed shape in model:', model.pos_embed.shape)
    # sys.exit()
    

    print('Model loaded from transformers', model)
    # sys.exit()
    # for p in model.parameters():
    #     p.requires_grad = False
    
    # model.eval()
    # model.to(device)
    print('HERE')
    ## LINES TO OPEN DENSE IMAGES -----------
    # img = Image.open('image.png')
    # img = Image.open('cow-beach.jpg')
    # print(f'image size: {img.size}')
    # img0 = img.convert('RGB')
    #
    ## FOR LArTPC EVENTS -------------
    img0 = event
    ## ---------------------------------
    print(f'Converted image size: {img0.size}')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img0)
    print(f'Initial image shape : {img.shape}')

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    print(f'cropped size: {(w, h)}')

    img = img[:, :w, :h].unsqueeze(0)

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
    # print(f'Attention after reshape: {attentions.shape}')
    # sys.exit()

    ## apply mask
    # threshold = 0.3
    threshold = None
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # attentions = attentions[0, :, 0:].reshape(nh, -1)
    print(attentions.shape)
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1)) 
    attentions[:, 283] = 0 

    ##
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)
    
    plt.imsave(fname=os.path.join(output_dir, "event.png"), arr=img0, format='png')
    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".jpg")
        plt.imsave(fname=fname, arr=attentions[j], format='jpg')
        print(f"{fname} saved.")

    if threshold is not None:
        image = skimage.io.imread(os.path.join('.', "image.png"))
        print(f'Image shape : {image.shape}')
        image = np.asarray(image)
        print(f'Image shape after skimage.io.imread : {image.shape}')

        # make the image divisible by the patch size
        image = skimage.transform.resize(image, image_size + (image.shape[2],), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        # w, h = image.shape[0] - image.shape[0] % patch_size, image.shape[1] - image.shape[1] % patch_size
        w, h = image_size[0] - image_size[0] % patch_size, image_size[1] - image_size[1] % patch_size
        print(f'cropped size: {(w, h)}')

        image = image[:w, :h, :]
        
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(output_dir, "mask_th" + str(threshold) + "_head" + str(j) +".jpg"), blur=False)

