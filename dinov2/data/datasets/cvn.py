import os, zlib, numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import sys

# ## Split cvn dataset
# ## ADD SPLITTING FUNCTIONALITY
# from typing import Union
# from enum import Enum

# _Target = int

# class _Split(Enum):
#     TRAIN       = "train"
#     VAL         = "val"
#     TEST        = "test"
    
#     @property
#     def length(self) -> int:
#         split_lengths = {
#             _Split.TRAIN: 100_000,
#             _Split.VAL: 20_000,
#             _Split.TEST: 20_000,
#         }
#         return split_lengths[self]

## Read info file for event ::: function from Nitish's repository dune_cvn.ipynb
def get_eventinfo(info_path):
    path = info_path
    ret = {}
    with open(path, 'rb') as info_file:
        info = info_file.readlines()
        ret['NuPDG'] = int(info[7].strip())
        ret['NuEnergy'] = float(info[1])
        ret['LepEnergy'] = float(info[2])
        ret['Interaction'] = int(info[0].strip()) % 4
        ret['NProton'] = int(info[8].strip())
        ret['NPion'] = int(info[9].strip())
        ret['NPiZero'] = int(info[10].strip())
        ret['NNeutron'] = int(info[11].strip())
        #ret['OscWeight'] = float(info[6])
    return ret

###-----------------------------------
class CVNDataset(Dataset):
    """
    ViT transformer expect 3 channels RGB-like, the input should fit this scheme
    DUNE CVN pixel-maps as 3-channel images (U,V,Z) from .gz files.
    Also we can do a single plane in 3 channels so it would be training on 1 plane instead of 3
    Returns (image, dummy_target). DINOv2 will apply its own multi-crop transform.
    """
    # # add global types
    # Target = Union[_Target]
    # Split = Union[_Split]

    def __init__(self, root,
                 split=None,
                # split: "CVNDataset.Split",
                 swap_axes=False,          
                 transform=None,
                 target_transform=None,
                 **kwargs):
        self.root = root
        self.split = split
        self.swap_axes = bool(swap_axes)
        self.transform = transform
        self.target_transform = target_transform

        # -------------------------------
        # Parse optional flags from :extra=
        # -------------------------------
        # Accept comma-separated K=V pairs, e.g. extra="plane=Z,mono=mono3,swap_axes=1"
        self.plane = None       # 'U','V','Z','0','1','2' -> choose single plane; None -> use all 3
        self.mono = "mono3"     # 'mono3' (replicate single plane to 3ch) | 'mono1' (true 1ch 'L' image), mono1 is actually not implemented, it requires changing the model which I think will be done anyway, so just keep config for now
        extra = kwargs.get("extra", None)
        if isinstance(extra, str):
            for token in extra.split(","):
                if not token.strip():
                    continue
                k, _, v = token.partition("=")
                k = k.strip().lower()
                v = v.strip()
                if k == "plane":
                    self.plane = v.upper()  # allow 'U','V','Z' or '0','1','2'
                elif k == "mono":
                    self.mono = v.lower()   # 'mono3' or 'mono1'
                elif k == "swap_axes":
                    self.swap_axes = v.lower() in ("1", "true", "t", "yes", "y")

        # -------------------------------
        # Index all available events
        # -------------------------------
        self.cand_flavs = ["nue", "numu", "NC", "nutau", "nuecc", "numucc", "nutaucc"]
        self.entries = []
        for flav in self.cand_flavs:
            d = os.path.join(root, flav)
            if not os.path.isdir(d):
                continue
            for gz in sorted(glob(os.path.join(d, "event*.gz"))):
                key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
                self.entries.append((flav, key, gz))
        # cand_flavs = ["nu", "nue", "nutau"]
        # self.entries = []
        # for flav in cand_flavs:
        #     folder_name = f'prodgenie_dunevd_1x8x6_{flav}/cvn_gaushit'
        #     d = os.path.join(root, folder_name)
        #     if not os.path.isdir(d):
        #         continue
        #     for dd in os.listdir(d):
        #         subdir = os.path.join(d, dd)
        #         if not os.path.isdir(subdir):
        #             continue
        #         # print(subdir)
        #         for gz in sorted(glob(os.path.join(subdir, "event*.gz"))):
        #             key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
        #             self.entries.append((flav, key, gz)) # since self.__getitem__ only uses gz, we can include the .info here and 
        #                                                                                             # 1) ignore it for test, 
        #                                                                                             # 2) return it to select specific information,
        #                                                 ## it looks like the info we need should be assigned to the target variable.
        # print("=============================== LENGTH OF DATASET : -------========" )
        # print(len(self.entries))
        # print(self.entries[0])
        # print('================================')
        # print('================================')
        # sys.exit()

        if not self.entries:
            raise RuntimeError(f"No .gz files found under {root}/<flavor>/event*.gz")
    
    def classes(self):
        return self.cand_flavs
    # ## get access to the self._split property
    # @property
    # def split(self) -> "CVNDataset.Split":
    #     return self._split
    
    def __len__(self):
        return len(self.entries)

    def _read_array(self, gz_path):
        with open(gz_path, "rb") as f:
            arr = np.frombuffer(bytearray(zlib.decompress(f.read())), dtype=np.uint8).reshape(3, 500, 500)
        if self.swap_axes:
            # swap wire/time -> (3, W, T) -> (3, T, W)
            arr = arr.transpose(0, 2, 1)
        return arr  # (3, H, W)

    def _to_pil(self, arr3):
        """
        Convert a (3,H,W) numpy uint8 to PIL according to options:
          - plane=None: return RGB-like from [U,V,Z] (3ch)
          - plane=..., mono=mono3: replicate selected plane -> 3ch RGB-like
          - plane=..., mono=mono1: return single-channel 'L' image (requires in_chans=1 model)
        """
        if self.plane is None:
            img = np.moveaxis(arr3, 0, -1)                     # (H,W,3)
            return Image.fromarray(img, mode="RGB")
        # choose one plane
        idx_map = {"U": 0, "V": 1, "Z": 2, "0": 0, "1": 1, "2": 2}
        idx = idx_map[self.plane]
        plane = arr3[idx]                                      # (H,W)
        if self.mono == "mono1":
            return Image.fromarray(plane, mode="L")            # true 1-channel
        # default: replicate into 3 channels (Option A)
        img = np.repeat(plane[..., None], 3, axis=2)           # (H,W,3)
        return Image.fromarray(img, mode="RGB")

    def __getitem__(self, idx):
        _, _, gz = self.entries[idx]
        arr = self._read_array(gz)
        img = self._to_pil(arr)
        # target = 0  # dummy label for SSL
        target = get_eventinfo(gz.replace('.gz', '.info'))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(f"TARGET VALUE : {target}")
        ## Try to predict flavor first. Later we can try to predict other things from the info file.
        target = self.cand_flavs.index(self.entries[idx][0])  # convert flavor to index
        return img, target
    


