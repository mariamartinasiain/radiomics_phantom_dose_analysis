import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from scipy.stats import ttest_rel

from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, EnsureTyped
from monai.data import SmartCacheDataset, DataLoader
from monai.losses import DiceCELoss
from swin_contrastive.swinunetr import  custom_collate_fn, get_model,load_data
from swin_contrastive.train import Train
from monai.config import print_config
import torch



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
    #qq chose comme testload = DataLoader(da.....

    dataset.start()
    i=0
    iterator = iter(dataload)
    for i,model in enumerate(models):
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        losses[i].append(mean_dice_val)   
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



def compare(jsonpath="./dataset_forgetting_test.json"):
    print_config()
    model1 = get_model(model_path = path1)
    model2 = get_model(model_path = path2)
    # trainer= Train(model, data_loader, optimizer, lr_scheduler, 40,dataset,contrastive_latentsize=700,savename="rois_ortho_0001_regularized.pth",ortho_reg=0.001)
    #trainer.train()
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
    
    data_dir = "data/"
    split_json = "dataset_0.json"

    datasets = data_dir + split_json
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = SmartCacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)
    
    data_loader = {"train": train_loader, "val": val_loader}
    dataset = {"train": train_ds, "val": val_ds}
    optimizer = torch.optim.Adam(model1.parameters(), 1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    t1 = Train(model1, data_loader, optimizer, lr_scheduler, 200,dataset,savename="baseline_segmentation.pth")
    t2 = Train(model2, data_loader, optimizer, lr_scheduler, 200,dataset,savename="finetuned_segmentation.pth")
    
    print("Training Baseline Model")
    t1.train()
    
    print("Training Finetuned Model")
    t2.train()
    
    model1 = get_model(model_path = "baseline_segmentation.pth")
    model2 = get_model(model_path = "finetuned_segmentation.pth")
    
    #model1 is the base model and model2 is the finetuned model to be compared
    models = [model1,model2]
    
    losses = run_testing(models,jsonpath,val_ds,val_loader)
    compare_losses(losses)