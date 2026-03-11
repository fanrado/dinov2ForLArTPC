import os, zlib, numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import sys
from torchvision import transforms
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
                 classification_type=None, ## Add classification_type argument to select the output target variable for from the dataset
                 **kwargs):
        self.root = root
        self.split = split
        self.swap_axes = bool(swap_axes)
        self.transform = transform
        self.target_transform = target_transform
        self.classification_type = classification_type ## Store classification_type for later use in __getitem__
        ### classification_type can be : 'flavor_2' for numu vs nue, 'flavor_3' for numu vs nue vs nc, 'ntracks' for (0, 1, 2, 3+) tracks, 'nshowers' for (0, 1, 2, 3+) showers.

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
        # self.cand_flavs = ["nue", "numu", "NC"] ## "NC" changed to "nc" to match folder name
        # self.cand_flavs = ["numu", "nue", "nc"]
        # self.entries = []
        # for flav in self.cand_flavs:
        #     d = os.path.join(root, flav)
        #     if not os.path.isdir(d):
        #         continue
        #     for gz in sorted(glob(os.path.join(d, "event*.gz"))):
        #         key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
        #         self.entries.append((flav, key, gz))

        # # self.pdgs = [12, 14, 16, -12, -14, -16, 1]  # corresponding PDG codes for flavors
        # # self.classes = ['numuCC', 'nueCC', 'nc']
        # self.classes = self.cand_flavs

        self.cand_flavs = ["nu", "nue"]#, "nutau"]
        self.entries = []
        for flav in self.cand_flavs:
            folder_name = f'prodgenie_dunevd_1x8x6_{flav}/cvn_gaushit'
            d = os.path.join(root, folder_name)
            if not os.path.isdir(d):
                continue
            for dd in os.listdir(d):
                subdir = os.path.join(d, dd)
                if not os.path.isdir(subdir):
                    continue
                # print(subdir)
                for gz in sorted(glob(os.path.join(subdir, "event*.gz"))):
                    key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
                    target = self.get_eventinfo(gz.replace('.gz', '.info'))
                    nuPDG = target["NuPDG"]
                    if nuPDG == -1:
                        continue  # skip unrecognized
                    ##--
                    ## Uncomment this block if you want to do ntracks or nshowers classification instead of flavor classification. For ntracks classification, we will skip nue events and only classify numuCC events based on their number of tracks. For nshowers classification, we will skip numu events and only classify nueCC events based on their number of showers. For flavor_3 classification, we will include all events and classify them into numu, nue, and nc based on their PDG code.
                    if self.classification_type == 'ntracks':
                        if nuPDG == 1: # skip nue, select numuCC only
                            continue
                    elif self.classification_type == 'nshowers':
                        if nuPDG == 0: # skip numu, select nueCC only
                            continue
                    ##--
                    if self.classification_type != 'flavor_3':
                        if nuPDG == 2:  # skip nc 
                            continue
                    self.entries.append((nuPDG, target['ntracks'], target['nshowers'], key, gz)) # since self.__getitem__ only uses gz, we can include the .info here and 
        # self.classes = self.cand_flavs
        # self.classes.append('nc')
        if self.classification_type == 'ntracks':
            self.classes = ['0', '1', '2', '3+']
        elif self.classification_type == 'nshowers':
            self.classes = ['0', '1', '2', '3+']
        elif self.classification_type == 'flavor_2':
            self.classes = ['numu', 'nue']#, 'nc']
        elif self.classification_type == 'flavor_3':
            self.classes = ['numu', 'nue', 'nc']
 
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
        img = np.repeat(plane[..., None], 3, axis=2)     # (H,W,3)
        return Image.fromarray(img, mode="RGB")

    def _to_original_pil(self, arr3):
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
        # if self.mono == "mono1":
        return Image.fromarray(plane, mode="L")            # true 1-channel
        # # default: replicate into 3 channels (Option A)
        # img = np.repeat(plane[..., None], 3, axis=2)           # (H,W,3)
        # return Image.fromarray(img, mode="RGB")
    
    ## Read info file for event ::: function from Nitish's repository dune_cvn.ipynb
    def get_eventinfo(self, info_path):
        path = info_path
        ret = {}
        with open(path, 'rb') as info_file:
            info = info_file.readlines()
            nuPDG = abs(int(info[7].strip()))
            ret['NuPDG'] = -1
            if nuPDG == 14:
                ret['NuPDG'] = 0
            if nuPDG == 12:
                ret['NuPDG'] = 1
            if nuPDG == 1:
                ret['NuPDG'] = 2
            ret['NuEnergy'] = float(info[1])
            ret['LepEnergy'] = float(info[2])
            ret['Interaction'] = int(info[0].strip()) % 4
            ret['NProton'] = int(info[8].strip())
            ret['NPion'] = int(info[9].strip())
            ret['NPiZero'] = int(info[10].strip())
            ret['NNeutron'] = int(info[11].strip())
            ## Add ntracks and nshowers which are commonly used for CVN classification tasks. For now, not include neutrons
            ntracks = 0
            nshowers = 0
            if nuPDG == 14:
                ntracks += 1
            if nuPDG == 12:
                nshowers += 1
            ntracks += int(info[8].strip())  # NProton
            ntracks += int(info[9].strip())  # NPion
            nshowers += 2*int(info[10].strip())  # NPiZero : 2 showers per pi0
            ret['ntracks'] = ntracks
            ret['nshowers'] = nshowers
            ## Cap ntracks and nshowers at 3+ for classification purposes. The value 3 will represent 3 or more tracks/showers, which is a common practice in CVN classification tasks to avoid having too many classes with very few samples.
            if ret['ntracks'] >= 3:
                ret['ntracks'] = 3
            if ret['nshowers'] >= 3:
                ret['nshowers'] = 3
            #ret['OscWeight'] = float(info[6])
        return ret
    
    def __getitem__(self, idx):
        label, ntracks, nshowers, key, gz = self.entries[idx]
        arr = self._read_array(gz)
        img = self._to_pil(arr)

        # convert_to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # img_original = convert_to_tensor(self._to_original_pil(arr))
        target = label
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.classification_type == 'ntracks':
            return img, ntracks
        elif self.classification_type == 'nshowers':
            return img, nshowers
        return img, label # return original image for visualization
    


