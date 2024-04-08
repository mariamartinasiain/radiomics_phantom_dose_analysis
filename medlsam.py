import os
import sys


import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from monai.data import Dataset, DataLoader
from monai.transforms.transform import LazyTransform, MapTransform
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Resized, 
    ToTensord,Orientationd, 
    MaskIntensityd,
    Transform,
    EnsureChannelFirstd,
)
from MedLSAM.MedSAM.segment_anything.build_sam import sam_model_registry
from swinunetr import CropOnROId, load_data,jsonpath,DebugTransform
from qa4iqi_extraction.constants import (
    SERIES_NUMBER_FIELD,
    SERIES_DESCRIPTION_FIELD,
    MANUFACTURER_MODEL_NAME_FIELD,
    MANUFACTURER_FIELD,
    SLICE_SPACING_FIELD,
    SLICE_THICKNESS_FIELD,
)

nettype = "vit_b" #encoder_embed_dim=768
vit_load_path = "sam_vit_b_01ec64.pth"

medsam = sam_model_registry[nettype](checkpoint=vit_load_path).to('cuda:0')
encoder = medsam.image_encoder

#on peut pas réutiliser le transform pour swinunetr, il faudra refaire un transform avec du pooling (sam prend du 2D) et du resize (taille de l'image = 1024)

class AveragePoolingDepthd(MapTransform):
    """
    Une transformation pour appliquer l'average pooling sur la dimension de profondeur
    des images dans les clés spécifiées d'un dictionnaire.
    """

    def __init__(self, keys):
        """
        Initialiser la transformation.
        :param keys: les clés des images dans le dictionnaire auxquelles appliquer l'average pooling.
        """
        super().__init__(keys)

    def __call__(self, data):
        """
        Appliquer la transformation.
        :param data: le dictionnaire contenant les images.
        :return: le dictionnaire avec les images transformées.
        """
        d = dict(data)
        for key in self.keys:
            # Assurez-vous que l'image est un tensor avant de faire l'average pooling
            d[key] = torch.mean(d[key], dim=1, keepdim=False)
        return d


transforms = Compose([
    LoadImaged(keys=["image", "roi"]),
    DebugTransform(),
    EnsureChannelFirstd(keys=["image", "roi"]),
    CropOnROId(keys=["image"], roi_key="roi",size=(64,64,64)), 
    DebugTransform(),  # Check the shape right after resizing
    #MaskIntensityd(keys=["image"], mask_key="roi"),
    AveragePoolingDepthd(keys=["image"]),
    DebugTransform(),
    Resized(spatial_size=(1024, 1024), mode='bilinear', keys=["image"]),
    DebugTransform(),
    ToTensord(keys=["image", "roi"]),
    #Orientationd(keys=["image", "roi"], axcodes="RAS"),
])

datafiles = load_data(jsonpath)
dataset = Dataset(data=datafiles, transform=transforms)
dataload = DataLoader(dataset, batch_size=1)

slice_num = 50
csv_data = []

for batch in dataload:
    image = batch["image"]
    input_image = medsam.preprocess(image[None,:,:,:])
    x_in = image.cuda()
    val_inputs = x_in.cuda()
    latentrep = encoder(val_inputs)
    print(latentrep.shape)
    record = {
            "SeriesNumber": batch["info"][SERIES_NUMBER_FIELD][0],
            "SeriesDescription": batch["info"][SERIES_DESCRIPTION_FIELD][0],
            "ManufacturerModelName" : batch["info"][MANUFACTURER_MODEL_NAME_FIELD][0],
            "Manufacturer" : batch["info"][MANUFACTURER_FIELD][0],
            "SliceThickness": batch["info"][SLICE_THICKNESS_FIELD][0],
            "SpacingBetweenSlices": batch["info"][SLICE_SPACING_FIELD][0],
            "ROI": batch["roi_label"][0],
            "deepfeatures": latentrep.flatten().tolist()  # Convertir en liste pour la sauvegarde CSV
    }
    csv_data.append(record)

df = pd.DataFrame(csv_data)
df.to_csv("deepfeaturesmedsam.csv", index=False)