import os
import shutil
import tempfile

import json
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms.utils import generate_spatial_bounding_box, compute_divisible_spatial_size,convert_data_type
from monai.transforms.transform import LazyTransform, MapTransform
from monai.utils import ensure_tuple,convert_to_tensor
from monai.transforms.croppad.array import Crop
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    Resized, 
    ToTensord,Orientationd, 
    MaskIntensityd,
    Transform,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

print_config()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DebugTransform(Transform):
    def __call__(self, data):
        # Print the shape of the image and the mask
        print(f"Image shape: {data['image'].shape}, Mask shape: {data['roi'].shape}")
        # Print unique values in the mask
        print(f"Unique values in mask: {np.unique(data['roi'])}")
        # Print the sum of the image pixel values (as an example metric)
        print(f"Sum of image pixel values: {data['image'].sum()}")
        return data

class CropOnROI(Crop):
    def compute_center(self, img: torch.Tensor):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        
        def is_positive(x):
            return x > 0
        
        box_start, box_end = generate_spatial_bounding_box(
            img, is_positive, None, 0, True
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=1))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        mid_point = np.floor((box_start_ + box_end_) / 2)
        print("MID POINT",mid_point)
        return mid_point
    
    def __init__(self, roi,size, lazy=False):
        super().__init__(lazy)
        center = self.compute_center(roi)
        self.slices = self.compute_slices(
            roi_center=center, roi_size=size, roi_start=None, roi_end=None, roi_slices=None
        )
    def __call__(self, img: torch.Tensor, lazy = None):
        lazy_ = self.lazy if lazy is None else lazy
        return super().__call__(img=img, slices=ensure_tuple(self.slices), lazy=lazy_)
        
class CropOnROId(MapTransform, LazyTransform):
    backend = Crop.backend

    def __init__(self, keys,roi_key,size, allow_missing_keys: bool = False, lazy: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.roi_key = roi_key
        self.size = size

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        if isinstance(self.cropper, LazyTransform):
            self.cropper.lazy = value


    def __call__(self, data, lazy= None):
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        print("LA SHAPE DE SIZE",(torch.tensor(self.size)).shape)
        for key in self.key_iterator(d):
            d[key] = CropOnROI(d[self.roi_key],size=self.size,lazy=lazy_)(d[key])
        return d
    
jsonpath = "./dataset_info.json"
def load_data(datalist_json_path):
  with open(datalist_json_path, 'r') as f:
          datalist = json.load(f)
  return datalist


target_size = (96, 96, 96)

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

weight = torch.load("./model_swinvit.pt")
model.load_from(weights=weight)
model = model.to('cuda')
print("Using pretrained self-supervied Swin UNETR backbone weights !")

transforms = Compose([
    LoadImaged(keys=["image", "roi"]),
    DebugTransform(),
    CropOnROId(keys=["image"], roi_key="roi",size=target_size), 
    DebugTransform(),  # Check the shape right after resizing
    #MaskIntensityd(keys=["image"], mask_key="roi"),
    ToTensord(keys=["image", "roi"]),
    Orientationd(keys=["image", "roi"], axcodes="RAS"),
])

datafiles = load_data(jsonpath)
dataset = Dataset(data=datafiles, transform=transforms)
dataload = DataLoader(dataset, batch_size=1)

slice_num = 50

for batch in dataload:
  #plutot for roi in batch[rois] ...
  image = batch["image"]
  x_in = image.cuda()
  val_inputs = torch.unsqueeze(x_in, 1).cuda()

  fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 6))
  ax1.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num], cmap="gray")
  ax1.set_title('Image')
  #ax3.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num])
  #ax3.set_title(f'Predict')
  plt.show()

  val_outputs = model.swinViT(val_inputs)
  latentrep = val_outputs[4] #48*2^4 = 768
  latentrep = model.encoder10(latentrep)
  print(latentrep.shape)