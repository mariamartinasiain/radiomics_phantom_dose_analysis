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
from monai.config import print_config
import torch



def run_testing(models,jsonpath = "./dataset_forgetting_test.json"):
    print_config()
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
    ])

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    losses = [[] for _ in models]
    
    datafiles = load_data(jsonpath)
    dataset = SmartCacheDataset(data=datafiles, transform=transforms, cache_rate=1, progress=True, num_init_workers=8, num_replace_workers=8)
    print("dataset length: ", len(datafiles))
    dataload = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=4)
    #qq chose comme testload = DataLoader(da.....

    dataset.start()
    i=0
    iterator = iter(dataload)
    for _ in tqdm(range(len(datafiles))):
        batch = next(iterator)               
        image,y = (batch["image"],batch["label"].cuda())
        val_inputs = image.cuda()
        print(val_inputs.shape)
        for i,model in enumerate(models):
            logit_map = model(val_inputs)
            loss = loss_function(logit_map, y)
            losses[i].append(loss.item())
            
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
    model1 = get_model(path1)
    model2 = get_model(path2)
    #model1 is the base model and model2 is the finetuned model to be compared
    models = [model1,model2]
    losses = run_testing(models,jsonpath)
    compare_losses(losses)