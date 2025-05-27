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
import tensorflow as tf
import subprocess
import json
import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    EnsureTyped,
    SaveImaged,
    ConcatItemsd,
    SplitDimd,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
import torch
from tqdm import tqdm

import os
import json
import torch
import SimpleITK as sitk
import pydicom
import numpy as np
from monai.transforms import Compose, EnsureTyped
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
import uuid
import datetime
from tqdm import tqdm
import json
import random

def load_subbox_positions(filename, order='XYZ', num_positions=None, seed=42):
    """
    Loads subbox positions from a file and optionally shuffles and limits the number of positions returned.
    Args:
        filename (str): Path to the file containing positions, supports .npy or .json.
        order (str): Order of dimensions, 'XYZ' or 'ZYX'.
        num_positions (int, optional): Number of positions to return, if None returns all.
        seed (int): Seed for the random number generator for shuffling positions.
    Returns:
        list: A list of tuples representing the loaded and possibly reordered positions.
    """
    # Chargement des positions
    if filename.endswith('.npy'):
        positions = np.load(filename)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            positions = json.load(f)
    else:
        raise ValueError("Format de fichier non supporté. Utilisez .npy ou .json")
    
    # Sélection aléatoire des positions si num_positions est spécifié
    if num_positions is not None:
        random.seed(seed)
        positions = random.sample(list(positions), min(num_positions, len(positions)))
    
    if order.upper() == 'XYZ':
        return positions
    elif order.upper() == 'ZYX':
        return [(p[2], p[1], p[0]) for p in positions]
    else:
        raise ValueError("Ordre non supporté. Utilisez 'XYZ' ou 'ZYX'")

def load_forbidden_boxes(filename):
    """
    Loads forbidden box positions from a file.
    Args:
        filename (str): Path to the file containing forbidden positions.
    Returns:
        list: A list of tuples, where each tuple contains the position and size of a forbidden box.
    """
    forbidden_boxes = []
    with open(filename, 'r') as f:
        for line in f:
            pos = [int(x) for x in line.strip().split(',')]
            forbidden_boxes.append((pos, [64, 64, 32]))  # Ajout de la taille fixe
    return forbidden_boxes

def sample_subboxes(box_list, big_box_size, subbox_size, num_samples, constraint_box):
    """
    Samples valid subbox positions given constraints such as other box positions and size limits.
    Args:
        box_list (list): List of boxes with positions and sizes that new boxes shouldn't overlap.
        big_box_size (list): Size of the big box within which subboxes are sampled.
        subbox_size (list): Desired size of each subbox.
        num_samples (int): Number of valid subbox positions to sample.
        constraint_box (list): Defines the region within the big box where subboxes can be placed.
    Returns:
        list: List of valid subbox positions.
    """
    def overlaps(pos, size):
        for box in box_list:
            b_pos, b_size = box
            if all(p < b_p + b_s and b_p < p + s for p, b_p, s, b_s in zip(pos, b_pos, size, b_size)):
                return True
        return False

    def inside_constraint_box(pos):
        return (constraint_box[0] <= pos[2] < constraint_box[1] and
                constraint_box[2] <= pos[0] < constraint_box[3] and
                constraint_box[4] <= pos[1] < constraint_box[5])

    valid_positions = []
    attempts = 0
    max_attempts = num_samples * 100

    while len(valid_positions) < num_samples and attempts < max_attempts:
        pos = [
            random.randint(constraint_box[2], constraint_box[3] - subbox_size[0]),
            random.randint(constraint_box[4], constraint_box[5] - subbox_size[1]),
            random.randint(constraint_box[0], constraint_box[1] - subbox_size[2])
        ]
        
        if not overlaps(pos, subbox_size) and inside_constraint_box(pos):
            valid_positions.append(pos)
        
        attempts += 1

    return valid_positions

def sample_and_save_subboxes(box_list, big_box_size, subbox_size, num_samples, output_dir, filename_prefix):
    """
    Samples valid subbox positions and saves them to a JSON file.
    Args:
        box_list (list): List of existing boxes to avoid.
        big_box_size (list): Size of the parent box.
        subbox_size (list): Desired size of each subbox.
        num_samples (int): Number of samples to generate.
        output_dir (str): Directory where output JSON file is saved.
        filename_prefix (str): Prefix for the filename of the output JSON.
    Returns:
        str: Filename of the saved JSON file containing the valid positions.
    """
    valid_positions = sample_subboxes(box_list, big_box_size, subbox_size, num_samples)

    os.makedirs(output_dir, exist_ok=True)

    json_filename = os.path.join(output_dir, f"{filename_prefix}_positions.json")
    with open(json_filename, 'w') as f:
        json.dump(valid_positions, f)
    
    return json_filename

def process_forbidden_boxes_and_sample(forbidden_boxes_file, big_box_size, subbox_size, num_samples, output_dir, filename_prefix):
    """
    Processes forbidden boxes from a file, samples valid subboxes avoiding these, and saves the results.
    Args:
        forbidden_boxes_file (str): Path to the file containing the forbidden boxes.
        big_box_size (list): Size of the parent box.
        subbox_size (list): Desired size of the subboxes.
        num_samples (int): Number of samples to generate.
        output_dir (str): Directory to save the sampled positions.
        filename_prefix (str): Prefix for the filename of the output JSON.
    Returns:
        str: Filename of the output JSON file containing the positions.
    """
    # Charger les boîtes interdites
    forbidden_boxes = load_forbidden_boxes(forbidden_boxes_file)
    
    # Échantillonner et sauvegarder les sous-boîtes valides
    json_filename = sample_and_save_subboxes(forbidden_boxes, big_box_size, subbox_size, num_samples, output_dir, filename_prefix)
    
    return json_filename

def nload_from(model, weights):
    """
    Loads weights into a model with detailed logging of updated and total weights.
    Args:
        model (torch.nn.Module): The model to load weights into.
        weights (dict): A dictionary containing the weights.
    Returns:
        tuple: Number of updated weights and the total weights processed.
    """
    updated_count = 0
    total_count = 0

    def update_and_count(param, weight_name):
        nonlocal updated_count, total_count
        if weight_name in weights:
            total_count += param.numel()
            if not torch.equal(param.data, weights[weight_name]):
                param.data.copy_(weights[weight_name])
                updated_count += param.numel()
        else:
            print(f"Warning: {weight_name} not found in weights")

    with torch.no_grad():
        update_and_count(model.swinViT.patch_embed.proj.weight, "swinViT.patch_embed.proj.weight")
        update_and_count(model.swinViT.patch_embed.proj.bias, "swinViT.patch_embed.proj.bias")
        
        for layer_name in ['layers1', 'layers2', 'layers3', 'layers4']:
            layer = getattr(model.swinViT, layer_name)
            for bname, block in layer[0].blocks.named_children():
                for name, param in block.named_parameters():
                    update_and_count(param, f"swinViT.{layer_name}.0.blocks.{bname}.{name}")
            
            update_and_count(layer[0].downsample.reduction.weight, f"swinViT.{layer_name}.0.downsample.reduction.weight")
            update_and_count(layer[0].downsample.norm.weight, f"swinViT.{layer_name}.0.downsample.norm.weight")
            update_and_count(layer[0].downsample.norm.bias, f"swinViT.{layer_name}.0.downsample.norm.bias")

    print(f"Updated {updated_count} out of {total_count} weights")
    return updated_count, total_count



def get_model(target_size=(64, 64, 32), model_path="model_swinvit.pt", to_compare=False):
    """
    Initializes and returns a model with optional comparison mode and specific target size.
    Args:
        target_size (tuple): Dimensions to initialize the model with.
        model_path (str): Path to the model's weights file.
        to_compare (bool): Whether to initialize the model for comparison purposes.
    Returns:
        torch.nn.Module: The initialized model.
    """
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if not to_compare:
        model = SwinUNETR(
            img_size=target_size,
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
    else:
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=14,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
    '''
    if "model_swinvit.pt" in model_path:
        weight = torch.load(model_path, weights_only=True)
        model.load_from(weight)
    else:
        weight = torch.load(model_path, weights_only=True)
        for key in weight.keys():
            print(key)
        weight = {k.replace("swinViT.","module."): v for k, v in weight.items()}
        weight = {k.replace("linear1.","fc1."): v for k, v in weight.items()}
        weight = {k.replace("linear2.","fc2."): v for k, v in weight.items()}
        for key in weight.keys():
            print("new : " + str(key))
        weight = {"state_dict" : weight}
        model.load_from(weight)

    # Cargar pesos
    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Extraer el state_dict correcto
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
        # Quitar "module." si es necesario
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    # Cargar pesos sin error aunque falten algunas claves
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded model with missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model = model.to(device)
    print("Model ready and loaded with pretrained weights.")
    return model

    '''
    print("Loaded weight keys:", weight.keys())
    model = model.to('cuda')
    print("Using pretrained self-supervied Swin UNETR backbone weights !")
    return model
    '''

def plot_multiple_losses(train_losses, step_interval):
    """
    Plots multiple types of losses across training steps.
    Args:
        train_losses (dict): Dictionary containing different types of losses.
        step_interval (int): Interval of steps to consider for plotting.
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    loss_types = ['contrast_losses', 'classification_losses', 'reconstruction_losses', 'orthogonality_losses', 'total_losses']
    
    for i, loss_type in enumerate(loss_types):
        losses = train_losses[loss_type]
        if losses:  # Check if the list is not empty
            points = len(losses)
            steps = np.arange(0, points * step_interval, step_interval)
            
            # Convert losses to numpy array, handling both tensor and float cases
            losses_np = np.array([loss.detach().cpu().numpy() if hasattr(loss, 'detach') else loss for loss in losses])
            
            row = i // 2
            col = i % 2
            axs[row, col].plot(steps, losses_np, label=f'{loss_type.capitalize()}')
            axs[row, col].set_title(f'{loss_type.capitalize()}')
            axs[row, col].set_xlabel('Steps')
            axs[row, col].set_ylabel('Loss')
            axs[row, col].legend()
    
    # Remove the unused subplot
    fig.delaxes(axs[2, 1])
    
    plt.tight_layout()
    unique_id = uuid.uuid4().hex[:8]  
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'losses_plot_{timestamp}_{unique_id}.png'
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free up memory

def convert_to_serializable(obj):
    """
    Converts objects to a format that can be serialized to JSON.
    Args:
        obj (any): The object to convert, which can be a list or torch.Tensor.
    Returns:
        any: A JSON-serializable representation of the object.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()  # Convert tensor to list
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_losses(train_losses, output_dir, to_compare=False):
    """
    Saves training losses to JSON files.
    Args:
        train_losses (dict): Dictionary containing different types of losses.
        output_dir (str): Directory to save loss data files.
        to_compare (bool): Flag to indicate whether to include comparison data.
    """
    #serializable_val_losses = convert_to_serializable(self.val_losses)
    serializable_contrast_losses = convert_to_serializable(train_losses['contrast_losses'])
    serializable_classification_losses = convert_to_serializable(train_losses['classification_losses'])
    serializable_total_losses = convert_to_serializable(train_losses['total_losses'])
    serializable_recosntruction_losses = convert_to_serializable(train_losses['reconstruction_losses'])
    #self.train_losses['orthogonality_losses'].append(self.losses_dict['orthogonality_loss'])
    serializable_orthogonality_losses = convert_to_serializable(train_losses['orthogonality_losses'])
    if to_compare:
        serializable_dice_losses = convert_to_serializable(train_losses['dice_losses'])
    
    # with open(loss_file, 'w') as f:
    #     json.dump({'train_losses': serializable_train_losses, 'val_losses': serializable_val_losses}, f)
    with open(f"{output_dir}_contrast_losses.json", 'w') as f:
        json.dump({'contrast_losses': serializable_contrast_losses}, f)
    with open(f"{output_dir}_classification_losses.json", 'w') as f:
        json.dump({'classification_losses': serializable_classification_losses}, f)
    with open(f"{output_dir}_total_losses.json", 'w') as f:
        json.dump({'total_losses': serializable_total_losses}, f)
    with open(f"{output_dir}_reconstruction_losses.json", 'w') as f:
        json.dump({'reconstruction_losses': serializable_recosntruction_losses}, f)
    with open(f"{output_dir}_orthogonality_losses.json", 'w') as f:
        json.dump({'orthogonality_losses': serializable_orthogonality_losses}, f)
    if to_compare:
        with open(f"{output_dir}_dice_losses.json", 'w') as f:
            json.dump({'dice_losses': serializable_dice_losses}, f)
    
import os
import glob
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImaged, SpatialCropd, EnsureTyped, EnsureChannelFirstd
from monai.data import DataLoader, Dataset

def crop_and_save_batch(input_folder, output_folder, crop_box, output_prefix):
    """
    Load a batch of medical images, crop each image, and save the results with a prefixed name.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where cropped images will be saved.
        crop_box (list): List specifying the bounding box to crop [x1, x2, y1, y2, z1, z2].
        output_prefix (str): Prefix to add to the output file names.
    """
    # Find all .nii.gz files in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.nii.gz"))

    # Define MONAI transforms to load and crop images
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),  # Ensure the channel is first
        EnsureTyped(keys=["image"]),  # Convert to tensor
        SpatialCropd(keys=["image"], roi_start=[crop_box[0], crop_box[2], crop_box[4]], 
                     roi_end=[crop_box[1], crop_box[3], crop_box[5]])
    ])

    # Create a dataset and dataloader
    dataset = Dataset(data=[{"image": path} for path in image_paths], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    # Process each image
    for idx, batch in enumerate(dataloader):
        image_data = batch["image"][0]  # Access the first (and only) item in the batch
        
        print(f"Shape of cropped image: {image_data.shape}")
        
        # Convert to numpy array and save
        cropped_image = image_data.numpy()  # Convert to numpy array

        # Save using nibabel
        cropped_image_nifti = nib.Nifti1Image(cropped_image[0,:,:,:], np.eye(4))
        print(f"shape of nifti image: {cropped_image_nifti.shape}")
        
        base_name = os.path.basename(image_paths[idx])  # Use the index to get the original filename
        output_name = f"{output_prefix}_{base_name}"
        output_path = os.path.join(output_folder, output_name)

        nib.save(cropped_image_nifti, output_path)
        print(f"Cropped image saved as: {output_path}")

def maincrop():
    """
    Main function to execute the cropping and saving of images in batch mode.
    """
    input_folder = "/mnt/nas7/data/reza/registered_dataset_pad/"
    output_folder = "cropped_liver/"
    crop_box = [67, 444, 120, 394, 130, 200]  
    output_prefix = "croppedliver"
    os.makedirs(output_folder, exist_ok=True)

    crop_and_save_batch(input_folder, output_folder, crop_box, output_prefix)


def load_data(datalist_json_path):
    """
    Loads data from a JSON file.
    Args:
        datalist_json_path (str): Path to the JSON file containing the data list.
    Returns:
        dict: A dictionary of the data list loaded from the JSON file.
    """
    with open(datalist_json_path, 'r') as f:
            datalist = json.load(f)
    return datalist


def load_data2(datalist_json_path):
    """
    Loads data from a JSON file and reformats it for processing.
    Args:
        datalist_json_path (str): Path to the JSON file containing the data list.
    Returns:
        list: A reformatted list of dictionaries for each data item.
    """
    with open(datalist_json_path, 'r') as f:
        datalist = json.load(f)
    return [{"image": v["image"], "roi": v["seg"]} for v in datalist.values()]

def setup_environment(device_id=0, output_dir="transformed_images"):
    """
    Sets up the computing environment by specifying the GPU and output directory.
    Args:
        device_id (int): GPU device ID.
        output_dir (str): Directory to save processed images.
    Returns:
        tuple: The device (torch.device) and output directory (str).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    return device, output_dir


# if __name__ == "__main__":
#     jsonpath = "merged_studies_map.json"  
#     registration(jsonpath)
    
def main_box():
    forbidden_boxes_file = "boxpos.txt"
    big_box_size = [512, 512, 343]  
    subbox_size = [64, 64, 32]  
    num_samples = 1000
    output_dir = "output"
    filename_prefix = "valid_positions2"
    constraint_box = [13, 323, 120, 395, 130, 200]  # z_start, z_end, x_start, x_end, y_start, y_end

    # Charger les boîtes interdites
    #forbidden_boxes = load_forbidden_boxes(forbidden_boxes_file)
    forbidden_boxes = []

    # Échantillonner et sauvegarder les sous-boîtes valides
    valid_positions = sample_subboxes(forbidden_boxes, big_box_size, subbox_size, num_samples, constraint_box)
    
    # Sauvegarder les positions valides
    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.join(output_dir, f"{filename_prefix}_positions.json")
    with open(json_filename, 'w') as f:
        json.dump(valid_positions, f)
    
    print(f"Les positions valides ont été sauvegardées dans : {json_filename}")
  
def resize_image(input_output):
    """
    Resizes an image using SimpleITK and saves it.
    Args:
        input_output (tuple): Contains input path, output path, and target size.
    Returns:
        bool: True if the image was successfully resized and saved, False otherwise.
    """
    input_path, output_path, target_size = input_output
    
    try:
        # Lire l'image
        image = sitk.ReadImage(input_path)
        
        # Obtenir les dimensions originales
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        
        # Calculer le nouveau spacing
        new_spacing = [
            (orig_sz * orig_spc) / targ_sz
            for orig_sz, orig_spc, targ_sz in zip(original_size, original_spacing, target_size)
        ]
        
        # Redimensionner l'image
        resample = sitk.ResampleImageFilter()
        resample.SetSize(target_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(image.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkLinear)
        
        resized_image = resample.Execute(image)
        
        # Sauvegarder l'image redimensionnée
        sitk.WriteImage(resized_image, output_path)
        return True
    except Exception as e:
        print(f"Erreur lors du traitement de {input_path}: {str(e)}")
        return False

def resize_and_save_images(json_path, output_dir, target_size=(512, 512, 343)):
    """
    Resizes and saves images specified in a JSON file.
    Args:
        json_path (str): Path to the JSON file containing image paths.
        output_dir (str): Directory to save the resized images.
        target_size (tuple): Target dimensions for resizing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données du JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Préparer les fichiers de données
    data = []
    for item in json_data:
        input_path = item['image']
        if not os.path.exists(input_path):
            print(f"Le fichier {input_path} n'existe pas. Il sera ignoré.")
            continue
        output_filename = os.path.basename(input_path).split('.')[0] + '_resized.nii.gz'
        output_path = os.path.join(output_dir, output_filename)
        data.append((input_path, output_path, target_size))
    
    # Traiter les images en parallèle
    with Pool() as pool:
        results = list(tqdm(pool.imap(resize_image, data), total=len(data)))
    
    successful = sum(results)
    print(f"Traitement terminé. {successful}/{len(data)} images ont été redimensionnées avec succès.")

import torch
import torch.nn as nn
class PyTorchModel(nn.Module):
    """
    A simple 3D convolutional PyTorch model copying the architecture of the tensorflow shallow cnn model
    """
    def __init__(self):
        super(PyTorchModel, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool3d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool3d(kernel_size=4, stride=4)
        
    def forward(self, x):
        # Input reshape
        x = x.view(-1, 64, 64, 32, 1)
        x = x.permute(0, 4, 1, 2, 3)  # Change to (N, C, D, H, W) format for PyTorch
        
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Flatten to 2048 features
        return x

def get_model_oscar(path):
    model = PyTorchModel()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()  
    return model

def convert_tf_to_pytorch():
    """
    Converts the shallow cnn TensorFlow model to a PyTorch model.
    Returns:
        PyTorchModel: The converted PyTorch model.
    """
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph('organs-5c-30fs-acc92-121.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    graph = tf.compat.v1.get_default_graph()
    
    pytorch_model = PyTorchModel()
    
    conv1_kernel = graph.get_tensor_by_name('Variable/read:0')
    conv1_bias = graph.get_tensor_by_name('Variable_1/read:0')
    conv2_kernel = graph.get_tensor_by_name('Variable_2/read:0')
    conv2_bias = graph.get_tensor_by_name('Variable_3/read:0')
    
    pytorch_model.conv1.weight.data = torch.DoubleTensor(sess.run(conv1_kernel).transpose(4, 3, 0, 1, 2))
    pytorch_model.conv1.bias.data = torch.DoubleTensor(sess.run(conv1_bias))
    pytorch_model.conv2.weight.data = torch.DoubleTensor(sess.run(conv2_kernel).transpose(4, 3, 0, 1, 2))
    pytorch_model.conv2.bias.data = torch.DoubleTensor(sess.run(conv2_bias))
    
    # Convertir tous les paramètres en double précision
    pytorch_model = pytorch_model.double()
    
    for param in pytorch_model.parameters():
        param.requires_grad = True
    
    
    return pytorch_model

def get_pytorch_model_for_inference():
    return convert_tf_to_pytorch()

def get_oscar_for_training():
    return convert_tf_to_pytorch()

if __name__ == "__main__":
    maincrop()
