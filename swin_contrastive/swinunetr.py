import csv
import os
import shutil
import tempfile

import json
import time
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from monai.data import Dataset, DataLoader,SmartCacheDataset
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms.utils import generate_spatial_bounding_box, compute_divisible_spatial_size,convert_data_type
from monai.transforms.transform import LazyTransform, MapTransform
from monai.utils import ensure_tuple,convert_to_tensor
from monai.transforms.croppad.array import Crop
from torch.utils.data._utils.collate import default_collate
from monai.transforms import (
    ScaleIntensityd,
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
    EnsureChannelFirstd,
    AsDiscreted,
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

from qa4iqi_extraction.constants import (
    SERIES_NUMBER_FIELD,
    SERIES_DESCRIPTION_FIELD,
    MANUFACTURER_MODEL_NAME_FIELD,
    MANUFACTURER_FIELD,
    SLICE_SPACING_FIELD,
    SLICE_THICKNESS_FIELD,
)


import torch






def filter_none(data, default_spacing=1.0):
    """Recursively filter out None values in the data and provide defaults for missing keys."""
    if isinstance(data, dict):
        filtered = {k: filter_none(v, default_spacing) for k, v in data.items() if v is not None}
        filtered['SpacingBetweenSlices'] = torch.tensor([default_spacing])
        return filtered
    elif isinstance(data, list):
        return [filter_none(item, default_spacing) for item in data if item is not None]
    return data

def custom_collate_fn(batch, default_spacing=1.0):
    filtered_batch = [filter_none(item, default_spacing) for item in batch]
    if not filtered_batch or all(item is None for item in filtered_batch):
        raise ValueError("Batch is empty after filtering out None values.")

    # Remove the ROI from the data to be collated, since it's not needed after image resizing
    for item in filtered_batch:
        if 'roi' in item:
            del item['roi']  

    try:
        return torch.utils.data.dataloader.default_collate(filtered_batch)
    except Exception as e:
        raise RuntimeError(f"Failed to collate batch: {str(e)}")


class DebugTransform(Transform):
    def __call__(self, data):
        print(f"Image shape: {data['image'].shape}, Mask shape: {data['roi'].shape}")
        print(f"Unique values in mask: {np.unique(data['roi'])}")
        print(f"Sum of image pixel values: {data['image'].sum()}")
        return data

class CropOnROI(Crop):
    def compute_center(self, img: torch.Tensor):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        
        def is_positive(x):
            return torch.gt(x, 0)
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
        #print("BOX START",box_start_)
        #print("BOX END",box_end_)
        #print("bouding box size",spatial_size)
        mid_point = np.floor((box_start_ + box_end_) / 2)
        #print("MID POINT",mid_point)
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

    def __init__(self, keys,roi_key,size, allow_missing_keys: bool = False, lazy: bool = False,id_key="id"):
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.id_key = id_key
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
        #print("LA SHAPE DE SIZE",(torch.tensor(self.size)).shape)
        for key in self.key_iterator(d):
            #print("KEY",key)
            d[key] = CropOnROI(d[self.roi_key],size=self.size,lazy=lazy_)(d[key])
            #d[self.id_key] = d['roi_label']
        return d

class CopyPathd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            data[f"{key}_path"] = data[key]  # Copier le chemin du fichier dans une nouvelle clé
        return data

def load_data(datalist_json_path):
        with open(datalist_json_path, 'r') as f:
                datalist = json.load(f)
        return datalist

def get_model(target_size = (64, 64, 32)):
    

    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = SwinUNETR(
        img_size=target_size,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    weight = torch.load("model_swinvit.pt")
    print("Loaded weight keys:", weight.keys())
    model.load_from(weight)
    #model.load_state_dict(weight)
    model = model.to('cuda')
    print("Using pretrained self-supervied Swin UNETR backbone weights !")
    return model

def run_inference(model,jsonpath = "./dataset_info_full_uncompressed_NAS_missing.json"):
    
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    print_config()
    target_size = (64, 64, 32)
    transforms = Compose([
        LoadImaged(keys=["image", "roi"], ensure_channel_first=True),
        CropOnROId(keys=["image"], roi_key="roi", size=target_size),
        CopyPathd(keys=["roi"]),
        EnsureTyped(keys=["image"], device=device, track_meta=False),
        
        #ToTensord(keys=["image"]),
    ])

    datafiles = load_data(jsonpath)
    #dataset = SmartCacheDataset(data=datafiles, transform=transforms, cache_rate=0.009, progress=True, num_init_workers=8, num_replace_workers=8)
    dataset = SmartCacheDataset(data=datafiles, transform=transforms,cache_rate=0.00049,progress=True,num_init_workers=8, num_replace_workers=8,replace_rate=0.2)
    dataset = SmartCacheDataset(data=datafiles, transform=transforms,cache_rate=0.049,progress=True,num_init_workers=8, num_replace_workers=8,replace_rate=0.2)
    print("dataset length: ", len(datafiles))
    dataload = ThreadDataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    #qq chose comme testload = DataLoader(da.....
    slice_num = 15
    with open("aaa.csv", "w", newline="") as csvfile:
        fieldnames = ["SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        dataset.start()
        i=0
        iterator = iter(dataload)
        for _ in tqdm(range(len(datafiles))):
            
            batch = next(iterator)               
            image = batch["image"]
            val_inputs = image#.cuda()
            print(val_inputs.shape)
            
            #val_outputs = model.swinViT(val_inputs)
            #latentrep = val_outputs[4] #48*2^4 = 768
            #latentrep = model.encoder10(latentrep)
            """print(latentrep.shape)
            record = {
                "SeriesNumber": batch["info"][SERIES_NUMBER_FIELD][0],
                "deepfeatures": latentrep.flatten().tolist(),
                "ROI": batch["roi_label"][0],
                "SeriesDescription": batch["info"][SERIES_DESCRIPTION_FIELD][0],
                "ManufacturerModelName" : batch["info"][MANUFACTURER_MODEL_NAME_FIELD][0],
                "Manufacturer" : batch["info"][MANUFACTURER_FIELD][0],
                "SliceThickness": batch["info"][SLICE_THICKNESS_FIELD][0],        
            }
            writer.writerow(record)"""
            #save 3d image
            print("Saving 3d image")
            image = image[0].cpu().numpy()
            image = np.squeeze(image)
            print("Image shape",image.shape)
            image = nib.Nifti1Image(image, np.eye(4))
            true_path = batch["roi_path"] 
            name = true_path[0]
            #remobing file path information and only keeping file name of the path
            name = os.path.basename(name)
            nib.save(image, "uncompress_cropped/"+name)
            
            
            if i%7 == 0:
                """print("Sleeping for 20 seconds")
                time.sleep(20)
                print("Woke up")"""
                dataset.update_cache()
                iterator = iter(dataload)
            i+=1
        dataset.shutdown()
        
    print("Done !")



def main():
    model = get_model()
    run_inference(model)

if __name__ == "__main__":
    main()