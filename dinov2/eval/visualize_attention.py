## Re-organize the script to be used in the training.
import os, zlib, sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_model(path_to_model: str, patch_size: int=14, image_size: tuple=(500, 500), model_type: str='vit_large'):
    """
    Docstring for load_model
    
    :param path_to_model: path to the model checkpoint file. The weights will be loaded from this file
        :type path_to_model: str
    :param patch_size: Patch size used in the Vision Transformer model
        :type patch_size: int
    :param image_size: Input image size (height, width)
        :type image_size: tuple
    :param model_type: Type of the Vision Transformer model. These models use the dino architecture. Options are: 'vit_small', 'vit_base', 'vit_large', 'vit_giant2
        :type model_type: str
    """
    from dinov2.models.vision_transformer import vit_large, vit_small, vit_base, vit_giant2, vit_tiny
    models = {
        'vit_tiny': vit_tiny,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_giant2': vit_giant2,
    }
    model = models[model_type](patch_size=patch_size, img_size=image_size[0])
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # disable gradients calculation
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    # Load model from a training checkpoint
    pth_model = torch.load(path_to_model)
    model.load_state_dict(pth_model, strict=False)
    return model

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

def get_attn(model: nn.Module, eval_gz_path: str=None, iteration: int=None, duringTraining=True, eval_output_dir: str=None, 
             sumoverheads: bool=False, image_size: int=500, patch_size: int=14, event=None):
    eval_output_dir = "attn/" if eval_output_dir is None else eval_output_dir

    if eval_gz_path is not None:
        event           = _read_array(gz_path=eval_gz_path)
        output_dir      = eval_output_dir + f'/{iteration:06d}/' if iteration is not None else eval_output_dir + '/eval/'
        if iteration is None:
            output_dir += os.path.basename(eval_gz_path).replace('.gz','')
    # image_size = (500, 500)
    # patch_size = 14
    # event           = _read_array(gz_path=eval_gz_path)
    # image_size      = tuple((image_size, image_size))
    patch_size      = patch_size

    device          = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for p in model.parameters():
         p.requires_grad = False
    model.eval()

    img0 = event
    print(f'img0 : {img0}')
    print(f'img0 type: {type(img0)}')
    img = None
    if eval_gz_path is not None:
        print(f'image size : {image_size}')
        transform = pth_transforms.Compose([
                pth_transforms.Resize(image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        img = transform(img0)
        print(f'type of img: {type(img)}')
        print(f'shape of img: {img.shape}')
    else:
        img = img0
    
    # print(f'type of img: {type(img)}')
    # print(f'shape of img: {img.shape}')

    # print(f"Transformed image size: {img.shape}")
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    print(f'img.shape after making divisible by patch size: {img.shape}')

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_self_attention(img.to(device))
    nh = attentions.shape[1] # number of head

    ## what the cls token is looking at in the input image
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(f'attentions shape after selecting cls token: {attentions.shape}')
    print(f'nh {nh}, w_featmap {w_featmap}, h_featmap {h_featmap}')

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    if eval_gz_path is None:
        print(f'attentions shape: {attentions.shape}')
        attention = np.sum(attentions, axis=0)
        return attention
    
    ## Get neutrino flavor from main folder and the filename as well.
    attn_sumheads_name = 'attn_sumheads.png'
    event_img_name = 'event.png'
    overlay_img_name = 'image_with_attention.png'
    print("----", eval_gz_path.split('/')[-4:])
    useful_name = '-'.join(eval_gz_path.split('/')[-4:]).replace('.gz','')
    event_img_name = useful_name + '-' + event_img_name
    attn_sumheads_name = useful_name + '-' + attn_sumheads_name
    overlay_img_name = useful_name + '-' + overlay_img_name

    # sum over heads
    if sumoverheads:
        # save attentions heatmaps
        os.makedirs(output_dir, exist_ok=True)
        attention = np.sum(attentions, axis=0)
        plt.imsave(fname=os.path.join(output_dir, event_img_name), arr=img0, format='png')
        plt.imsave(fname=os.path.join(output_dir, attn_sumheads_name), arr=attention, format='png')

        attention_mask_image = Image.open(os.path.join(output_dir, attn_sumheads_name)).convert("L").resize(img0.size)
        img0.paste(attention_mask_image, (0, 0), attention_mask_image)

        img0.save(f'{output_dir}/{overlay_img_name}')

        print(f"{os.path.join(output_dir, attn_sumheads_name)} saved.")

    else:
        # save attentions heatmaps
        os.makedirs(output_dir, exist_ok=True)
        plt.imsave(fname=os.path.join(output_dir, event_img_name), arr=img0, format='png')
        for j in range(nh):
            fname = os.path.join(output_dir, "attn-head" + str(j) + ".jpg")
            plt.imsave(fname=fname, arr=attentions[j], format='jpg')
            print(f"{fname} saved.")
    
    if duringTraining:
        # reactivate gradients calculation in the model
        for p in model.parameters():
            p.requires_grad = True

def visualize_attn_cls():
    parser = argparse.ArgumentParser(description='Visualize DINOv2 attn maps', add_help=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--eval_gz_path', type=str, required=True, help='Path to the eval gz file. This is a main folder that contains multiple gz files.')
    parser.add_argument('--patch_size', type=int, default=14, help='Patch size used in the Vision Transformer model')
    parser.add_argument('--image_size', type=int, nargs=2, default=(500, 500), help='Input image size (height, width)')
    parser.add_argument('--model_type', type=str, default='vit_large', help="Type of the Vision Transformer model. Options are: 'vit_small', 'vit_base', 'vit_large', 'vit_giant2'")
    parser.add_argument('--eval_output_dir', type=str, default=None, help='Directory to save the eval attention maps. If not provided, defaults to attn/')
    args = parser.parse_args()

    model = load_model(path_to_model=args.model_path, patch_size=args.patch_size, image_size=tuple(args.image_size), model_type=args.model_type)
    i = 0
    for gz_file in [f for f in os.listdir(args.eval_gz_path) if f.endswith('.gz')]:
        eval_gz_path = os.path.join(args.eval_gz_path, gz_file)
        print(f"Processing {eval_gz_path}...")
        get_attn(model=model, eval_gz_path=eval_gz_path, iteration=None, duringTraining=False, eval_output_dir=args.eval_output_dir, sumoverheads=True, image_size=tuple(args.image_size), patch_size=args.patch_size)
        # break  # Remove this break to process all files
        # if i == 2:
        #     break
        # i +=1

if __name__ == "__main__":
    visualize_attn_cls()