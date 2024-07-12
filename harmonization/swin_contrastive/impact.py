import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from scipy.stats import ttest_rel

import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Transform,
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

from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, EnsureTyped
from monai.data import SmartCacheDataset, DataLoader
from monai.losses import DiceCELoss
from harmonization.swin_contrastive.swinunetr import  custom_collate_fn, get_model,load_data
from harmonization.swin_contrastive.train import Train, get_device
from monai.config import print_config
import torch

def quick_weight_check(model1, model2, n_samples=100, seed=42):
    import random
    random.seed(seed)
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    
    if len(params1) != len(params2):
        return False  # Les modèles ont un nombre différent de couches

    sample_sum1 = 0
    sample_sum2 = 0
    
    for _ in range(n_samples):
        layer_idx = random.randint(0, len(params1) - 1)
        param1 = params1[layer_idx]
        param2 = params2[layer_idx]
        
        if param1.shape != param2.shape:
            return False  # Les couches ont des formes différentes
        
        idx = random.randint(0, param1.numel() - 1)
        sample_sum1 += param1.view(-1)[idx].item()
        sample_sum2 += param2.view(-1)[idx].item()
    
    return sample_sum1,sample_sum2

def run_testing(models,jsonpath = "./dataset_forgetting_test.json",val_ds=None,val_loader=None):
    torch.backends.cudnn.benchmark = True
    print_config()
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    losses = [[] for _ in models]
    dataset = val_ds
    dataload = val_loader
    dataset.start()

    s1,s2 = quick_weight_check(models[0], models[1])
    print(f"Weight Check: {s1} vs {s2}")
    
    epoch_iterator_val = tqdm(dataload, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    
    for i,model in enumerate(models):
        print(f"Testing Model {i+1}")
        j=0
        #print(f"Model: {model}")
        post_label = AsDiscrete(to_onehot=14)
        post_pred = AsDiscrete(argmax=True, to_onehot=14)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                print("j" , j)
                if j == 0:
                    import nibabel as nib
                    image = val_inputs.cpu().numpy()
                    label = val_labels.cpu().numpy()
                    pred = val_outputs.cpu().numpy()
                    nib.save(nib.Nifti1Image(image, np.eye(4)), str(i) + "image.nii.gz")
                    nib.save(nib.Nifti1Image(label, np.eye(4)), str(i) +"label.nii.gz")
                    nib.save(nib.Nifti1Image(pred, np.eye(4)), str(i) +"pred.nii.gz")

                j+=1
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                print(f"Dice: {dice_metric.aggregate().item()}")
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        losses[i].append(mean_dice_val)   
        
    dataset.shutdown()
    
    print("Done !")
    return losses

def compare_losses(losses,output_file="comparison_results.txt"):
    """
    Compare the losses of the baseline model and the finetuned model.
    
    Parameters:
    losses (list of list of float): A list containing two lists of losses,
                                    one for each model (baseline and finetuned).

    Returns:
    None
    """
    baseline_losses = losses[0]
    finetuned_losses = losses[1]
    baseline_losses = np.array(baseline_losses)
    finetuned_losses = np.array(finetuned_losses)


    baseline_mean = np.mean(baseline_losses)
    finetuned_mean = np.mean(finetuned_losses)
    baseline_std = np.std(baseline_losses)
    finetuned_std = np.std(finetuned_losses)

    print(f"Baseline Model - Mean Loss: {baseline_mean:.4f}, Std Dev: {baseline_std:.4f}")
    print(f"Finetuned Model - Mean Loss: {finetuned_mean:.4f}, Std Dev: {finetuned_std:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(baseline_losses, label='Baseline Model Losses', color='blue')
    plt.plot(finetuned_losses, label='Finetuned Model Losses', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Loss')
    plt.title('Comparison of Losses Between Baseline and Finetuned Models')
    plt.legend()
    plt.grid(True)
    plt.show()

    #difference in means
    mean_difference = finetuned_mean - baseline_mean
    print(f"Mean Difference (Finetuned - Baseline): {mean_difference:.4f}")

    #paired t-test
    t_stat, p_value = ttest_rel(baseline_losses, finetuned_losses)
    print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    with open(output_file, "w") as f:
        f.write(f"Baseline Model - Mean Loss: {baseline_mean:.4f}, Std Dev: {baseline_std:.4f}\n")
        f.write(f"Finetuned Model - Mean Loss: {finetuned_mean:.4f}, Std Dev: {finetuned_std:.4f}\n")
        f.write(f"Mean Difference (Finetuned - Baseline): {mean_difference:.4f}\n")
        f.write(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}\n")

class DebugPathTransform(Transform):
    def __call__(self, data):
        print(data)
        return data

def compare(jsonpath="./dataset_forgetting.json"):
    print_config()
    model1 = get_model(model_path = "model_swinvit.pt",to_compare=True) 
    model2 = get_model(model_path = "rois_contrastive_classif_ortho_0001_regularized.pth",to_compare=True)
    device = get_device()
    num_samples = 2
    # trainer= Train(model, data_loader, optimizer, lr_scheduler, 40,dataset,contrastive_latentsize=700,savename="rois_ortho_0001_regularized.pth",ortho_reg=0.001)
    #trainer.train()
    print("Loading Data")
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            DebugPathTransform(),
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )
    

    datasets = jsonpath
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = SmartCacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
    )
    print("Data Loaded")
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    val_ds = SmartCacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    print("Data Loaded")
    set_track_meta(False)
    
    
    print("Data Loade and Transformed")
    data_loader = {"train": train_loader, "test": val_loader}
    dataset = {"train": train_ds, "test": val_ds}
    optimizer = None
    lr_scheduler = None
    
    #t1 = Train(model1, data_loader, optimizer, lr_scheduler, 200,dataset,savename="baseline_segmentation3.pth",to_compare=True)
    #t2 = Train(model2, data_loader, optimizer, lr_scheduler, 200,dataset,savename="finetuned_segmentation3.pth",to_compare=True)
    
    #print("Training Baseline Model")
    #t1.train()
    
    #print("Training Finetuned Model")
    #t2.train()
    
    model1 = get_model(model_path = "baseline_segmentation3.pth",to_compare=True)
    model2 = get_model(model_path = "finetuned_segmentation3.pth",to_compare=True)
    
    #model1 is the base model and model2 is the finetuned model to be compared
    models = [model1,model2]
    
    losses = run_testing(models,jsonpath,val_ds,val_loader)
    compare_losses(losses)
    
if __name__ == "__main__":
    compare()