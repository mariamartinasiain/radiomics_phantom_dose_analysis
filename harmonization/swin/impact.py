#test the segmentation result (main task of the pretrained model) on both pretrained and finetuned model.
#inspired by this google colab : https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb
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
from harmonization.swin.extract import  custom_collate_fn, get_model,load_data
from harmonization.swin.train import get_device
from monai.config import print_config
import torch


import os
import torch
import numpy as np
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from tqdm import tqdm

class Train:
    def __init__(self, model, data_loader, optimizer, max_iterations, dataset, 
                 val_interval=500, device=None, savename="model.pth"):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.dataset = dataset
        self.val_interval = val_interval
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.savename = savename

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        global_step = 0

        while global_step < self.max_iterations:
            epoch_loss = 0
            step = 0
            self.model.train()
            epoch_iterator = tqdm(self.data_loader["train"], desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
            for step, batch in enumerate(epoch_iterator):
                step += 1
                x, y = (batch["image"].to(self.device), batch["label"].to(self.device))

                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = self.loss_function(logits, y)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, self.max_iterations, loss))

                if (global_step % self.val_interval == 0 and global_step != 0) or global_step == self.max_iterations:
                    epoch_loss /= step
                    epoch_loss_values.append(epoch_loss)
                    metric = self.validate()
                    metric_values.append(metric)

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = global_step
                        torch.save(self.model.state_dict(), self.savename)
                        print("saved new best metric model")

                    print(f"current epoch: {global_step} current mean dice: {metric:.4f}"
                          f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

                global_step += 1
                if global_step >= self.max_iterations:
                    break

            if global_step >= self.max_iterations:
                break

        print(f"train completed, best_metric: {best_metric:.4f} at iteration: {best_metric_epoch}")
        return epoch_loss_values, metric_values

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for val_data in self.data_loader["test"]:
                val_inputs, val_labels = (val_data["image"].to(self.device), val_data["label"].to(self.device))
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                self.dice_metric(y_pred=val_output_convert, y=val_labels_convert)

            metric = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
        return metric

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
    print_config()
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    losses = [[] for _ in models]
    dataset = val_ds
    dataload = val_loader
    #dataset.start()

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
                
                print(f"Shape of val_inputs: {val_inputs.shape}")
                print(f"Shape of val_labels: {val_labels.shape}")

                
                # with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                print(f"Shape of val_outputs: {val_outputs.shape}")
                print("j" , j)
                # if j == 0:
                #     import nibabel as nib
                #     img = val_inputs[0,:,:,:,:].cpu().numpy()
                #     img = np.squeeze(img)
                #     img = nib.Nifti1Image(img, np.eye(4))
                #     nib.save(img, str(i) + "image.nii.gz")
                    
                #     label = val_labels[0,:,:,:,:].cpu().numpy()
                #     label = np.squeeze(label)
                #     label = nib.Nifti1Image(label, np.eye(4))
                #     nib.save(label, str(i) + "label.nii.gz")                    
                    
                #     val_outputs_single_channel = torch.argmax(val_outputs, dim=1, keepdim=True)
                #     segmentation_map = val_outputs_single_channel.squeeze().cpu().numpy()
                #     segmentation_map = segmentation_map.astype(np.uint8)
                #     header = nib.Nifti1Header()
                #     header.set_data_dtype(np.uint8)
                #     nifti_img = nib.Nifti1Image(segmentation_map, affine=np.eye(4), header=header)
                #     pred = nib.Nifti1Image(nifti_img, np.eye(4))
                #     nib.save(pred, str(i) + "pred.nii.gz")
                    
            

                j+=1
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                
                print(f"Shape of val_labels_convert: {val_labels_convert[0].shape}")
                print(f"Shape of val_output_convert: {val_output_convert[0].shape}")
                
                
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                print(f"Dice: {dice_metric.aggregate().item()}")
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        losses[i].append(mean_dice_val)   
        
    #dataset.shutdown()
    
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
    torch.backends.cudnn.benchmark = True
    print_config()
    model1 = get_model(model_path = "model_swinvit.pt",to_compare=True) 
    model2 = get_model(model_path = "rois_contrastive_classif_ortho_0001_regularized.pth",to_compare=True)
    device = get_device()
    num_samples = 4
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
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    set_track_meta(False)
    
    data_loader = {"train": train_loader, "test": val_loader}
    dataset = {"train": train_ds, "test": val_ds}
    optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=1e-5)
    
    t1 = Train(model1, data_loader, optimizer, 50000, dataset, savename="baseline_segmentation6.pth")

    optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=1e-5)

    t2 = Train(model2, data_loader, optimizer, 50000, dataset, savename="finetuned_segmentation6.pth")
    
    #print("Training Baseline Model")
    baseline_loss_values, baseline_metric_values = t1.train()
    
    print("Training Finetuned Model")
    finetuned_loss_values, finetuned_metric_values = t2.train()
    
#     Error in LazyPatchLoader: Could not create IO object for reading file /mnt/nas7/data/reza/registered_dataset_pad/B2_195109_580000_SOMATOM_Edge_Plus_ID28_Harmonized_10mGy_FBP_NrFiles_343.nii.gz
# The file doesn't exist.
# Filename = /mnt/nas7/data/reza/registered_dataset_pad/B2_195109_580000_SOMATOM_Edge_Plus_ID28_Harmonized_10mGy_FBP_NrFiles_343.nii.gz
    model1 = get_model(model_path="baseline_segmentation6.pth", to_compare=True)
    model2 = get_model(model_path="finetuned_segmentation6.pth", to_compare=True)
    
    models = [model1, model2]
    
    losses = run_testing(models, jsonpath, val_ds, val_loader)
    compare_losses(losses)

    # Plot the training losses
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_loss_values, label='Baseline Model')
    plt.plot(finetuned_loss_values, label='Finetuned Model')
    plt.xlabel('Validation Interval')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('training_loss_comparison.png')
    plt.close()

    # Plot the validation metrics
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_metric_values, label='Baseline Model')
    plt.plot(finetuned_metric_values, label='Finetuned Model')
    plt.xlabel('Validation Interval')
    plt.ylabel('Dice Metric')
    plt.title('Validation Dice Metric')
    plt.legend()
    plt.savefig('validation_metric_comparison.png')
    plt.close()

if __name__ == "__main__":
    compare()