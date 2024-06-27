import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import SimpleITK as sitk
from monai.transforms import Compose, LoadImaged, EnsureTyped
from monai.data import SmartCacheDataset, ThreadDataLoader
from tqdm import tqdm

import os
import json
import torch
import SimpleITK as sitk
import pydicom
import numpy as np
from monai.transforms import Compose, EnsureTyped
from monai.data import Dataset, DataLoader
from tqdm import tqdm
import json


def plot_multiple_losses(train_losses, step_interval):
    """
    Plots the contrastive, classification, reconstruction, and total losses over the training steps.
    """
    
    step_interval = step_interval

    points = len(train_losses['contrast_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    contrast_losses = [loss.detach().numpy() for loss in train_losses['contrast_losses']]
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].plot(steps, contrast_losses, label='Contrastive Loss')
    ax[0, 0].set_title('Contrastive Loss')
    ax[0, 0].set_xlabel('Steps')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    points = len(train_losses['classification_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    classification_losses = [loss.detach().numpy() for loss in train_losses['classification_losses']]

    ax[0, 1].plot(steps, classification_losses, label='Classification Loss')
    ax[0, 1].set_title('Classification Loss')
    ax[0, 1].set_xlabel('Steps')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].legend()

    points = len(train_losses['reconstruction_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    reconstruction_losses = [loss.detach().numpy() for loss in train_losses['reconstruction_losses']]

    ax[1, 0].plot(steps, reconstruction_losses, label='Reconstruction Loss')
    ax[1, 0].set_title('Reconstruction Loss')
    ax[1, 0].set_xlabel('Steps')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 0].legend()

    points = len(train_losses['total_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    total_losses = [loss.detach().numpy() for loss in train_losses['total_losses']]

    ax[1, 1].plot(steps, total_losses, label='Total Loss')
    ax[1, 1].set_title('Total Loss')
    ax[1, 1].set_xlabel('Steps')
    ax[1, 1].set_ylabel('Loss')
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig('losses_plot.png')
    plt.show()
    
def load_data(datalist_json_path):
        with open(datalist_json_path, 'r') as f:
                datalist = json.load(f)
        return datalist


def load_data2(datalist_json_path):
    with open(datalist_json_path, 'r') as f:
        datalist = json.load(f)
    return [{"image": v["image"], "roi": v["seg"]} for v in datalist.values()]

def setup_environment(device_id=0, output_dir="transformed_images"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    return device, output_dir

def load_dicom_series(directory):
    dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
    
    slices = [pydicom.dcmread(f) for f in dicom_files]
    image = np.stack([s.pixel_array for s in slices])
    return image, slices

def calculate_transform(fixed_roi, moving_roi):
    fixed_image = sitk.GetImageFromArray(fixed_roi)
    moving_image = sitk.GetImageFromArray(moving_roi)
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    
    return registration_method.Execute(fixed_image, moving_image)

def apply_transform(image, transform, reference_image):
    sitk_image = sitk.GetImageFromArray(image)
    transformed_image = sitk.Resample(
        sitk_image, reference_image, transform, sitk.sitkLinear, 0.0, sitk_image.GetPixelID()
    )
    return sitk.GetArrayFromImage(transformed_image)

def save_dicom_series(transformed_image, original_slices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, slice in enumerate(original_slices):
        slice.PixelData = transformed_image[i].astype(np.uint16).tobytes()
        slice.save_as(os.path.join(output_dir, f"slice_{i:04d}.dcm"))

def process_images(dataloader, reference_roi, reference_image, device, output_dir):
    for idx, batch in enumerate(tqdm(dataloader)):
        image_dir, roi_path = batch["image"][0], batch["roi"][0]
        
        image, original_slices = load_dicom_series(image_dir)
        roi, _ = load_dicom_series(os.path.dirname(roi_path))
        
        transform = calculate_transform(reference_roi, roi)
        transformed_image = apply_transform(image, transform, reference_image)
        
        output_subdir = os.path.join(output_dir, f"transformed_image_{idx:04d}")
        save_dicom_series(transformed_image, original_slices, output_subdir)

def registration(jsonpath, device_id=0):
    device, output_dir = setup_environment(device_id)
    
    datafiles = load_data2(jsonpath) 
    dataset = Dataset(data=datafiles)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    
    reference_data = dataset[0]
    reference_roi, _ = load_dicom_series(os.path.dirname(reference_data["roi"]))
    reference_image, _ = load_dicom_series(reference_data["image"])
    reference_image = sitk.GetImageFromArray(reference_image)
    
    process_images(dataloader, reference_roi, reference_image, device, output_dir)
    
    print(f"All transformed images have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    jsonpath = "merged_studies_map.json"  
    registration(jsonpath)