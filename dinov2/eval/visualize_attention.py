## Re-organize the script to be used in the training.
import os, zlib, sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def _read_array(gz_path, swap_axes=False, plane='Z'):
        """
          Read a gzipped LArTPC image and return a PIL image in RGB format.
        """
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

def get_attn(model, iteration, eval_gz_path):
    eval_output_dir = "attn/"
    image_size = (500, 500)
    patch_size = 14
    event           = _read_array(gz_path=eval_gz_path)
    image_size      = tuple(image_size)
    output_dir      = eval_output_dir + f'/{iteration:06d}/'
    patch_size      = patch_size

    device          = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model           = vit_large(patch_size=patch_size, img_size=image_size[0])
    # model.to(device)
    # for p in model.parameters():
    #     p.requires_grad = False
    # model.eval()

    # pth_model = torch.load('../out_cvn_memlite_batchpergpu_16/model_final.rank_0.pth')
    # model.load_state_dict(pth_model, strict=False)
    for p in model.parameters():
         p.requires_grad = False
    model.eval()

    img0 = event
    transform = pth_transforms.Compose([
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    img = transform(img0)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_self_attention(img.to(device))
    nh = attentions.shape[1] # number of head

    ## what the cls token is looking at in the input image
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    plt.imsave(fname=os.path.join(output_dir, "event.png"), arr=img0, format='png')
    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".jpg")
        plt.imsave(fname=fname, arr=attentions[j], format='jpg')
        print(f"{fname} saved.")
    
    # reactivate gradients calculation in the model
    for p in model.parameters():
        p.requires_grad = True