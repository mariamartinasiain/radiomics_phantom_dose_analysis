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
from onnx2pytorch import ConvertModel
import onnx
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
    forbidden_boxes = []
    with open(filename, 'r') as f:
        for line in f:
            pos = [int(x) for x in line.strip().split(',')]
            forbidden_boxes.append((pos, [64, 64, 32]))  # Ajout de la taille fixe
    return forbidden_boxes

def sample_subboxes(box_list, big_box_size, subbox_size, num_samples, constraint_box):
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
    valid_positions = sample_subboxes(box_list, big_box_size, subbox_size, num_samples)

    os.makedirs(output_dir, exist_ok=True)

    json_filename = os.path.join(output_dir, f"{filename_prefix}_positions.json")
    with open(json_filename, 'w') as f:
        json.dump(valid_positions, f)
    
    return json_filename

def process_forbidden_boxes_and_sample(forbidden_boxes_file, big_box_size, subbox_size, num_samples, output_dir, filename_prefix):
    # Charger les boîtes interdites
    forbidden_boxes = load_forbidden_boxes(forbidden_boxes_file)
    
    # Échantillonner et sauvegarder les sous-boîtes valides
    json_filename = sample_and_save_subboxes(forbidden_boxes, big_box_size, subbox_size, num_samples, output_dir, filename_prefix)
    
    return json_filename

def nload_from(model, weights):
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



def get_model(target_size = (64, 64, 32),model_path = "model_swinvit.pt",to_compare=False):
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

    if model_path == "model_swinvit.pt":
        weight = torch.load("model_swinvit.pt")
        model.load_from(weight)
    else:
        weight = torch.load(model_path)
        for key in weight.keys():
            print(key)
        weight = {k.replace("swinViT.","module."): v for k, v in weight.items()}
        weight = {k.replace("linear1.","fc1."): v for k, v in weight.items()}
        weight = {k.replace("linear2.","fc2."): v for k, v in weight.items()}
        for key in weight.keys():
            print("new : " + str(key))
        weight = {"state_dict" : weight}
        model.load_from(weight)
        
        
    
    print("Loaded weight keys:", weight.keys())
    model = model.to('cuda')
    print("Using pretrained self-supervied Swin UNETR backbone weights !")
    return model

def plot_multiple_losses(train_losses, step_interval):
    """
    Plots the contrastive, classification, reconstruction, orthogonality, and total losses over the training steps.
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
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()  # Convert tensor to list
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_losses(train_losses, output_dir, to_compare=False):
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

def is_dicom(file_path):
    if os.path.isdir(file_path):
        return False
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except (pydicom.errors.InvalidDicomError, IsADirectoryError, PermissionError):
        return False

def load_dicom_series(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")
    
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_dicom(file_path):
                dicom_files.append(file_path)
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in directory or subdirectories: {directory}")
    
    def sort_key(f):
        try:
            return pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber
        except AttributeError:
            return f

    dicom_files.sort(key=sort_key)
    
    slices = [pydicom.dcmread(f) for f in dicom_files]
    image = np.stack([s.pixel_array for s in slices])
    return image, slices

def load_mask(file_path):
    mask = pydicom.dcmread(file_path).pixel_array
    return mask

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
        roi = load_mask(roi_path)
        
        transform = calculate_transform(reference_roi, roi)
        transformed_image = apply_transform(image, transform, reference_image)
        
        output_subdir = os.path.join(output_dir, f"transformed_image_{idx:04d}")
        save_dicom_series(transformed_image, original_slices, output_subdir)

def registration(jsonpath, device_id=0):
    device, output_dir = setup_environment(device_id)
    
    datafiles = load_data2(jsonpath) 
    if not datafiles:
        raise ValueError(f"No data found in JSON file: {jsonpath}")
    
    dataset = Dataset(data=datafiles)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    
    reference_data = dataset[0]
    try:
        reference_roi = load_mask(reference_data["roi"])
    except Exception as e:
        print(f"Error loading reference ROI: {e}")
        print(f"ROI path: {reference_data['roi']}")
        return

    try:
        reference_image, _ = load_dicom_series(reference_data["image"])
    except Exception as e:
        print(f"Error loading reference image: {e}")
        print(f"Image path: {reference_data['image']}")
        return

    reference_image_sitk = sitk.GetImageFromArray(reference_image)
    
    process_images(dataloader, reference_roi, reference_image_sitk, device, output_dir)
    
    print(f"All transformed images have been saved in the directory: {output_dir}")

# if __name__ == "__main__":
#     jsonpath = "merged_studies_map.json"  
#     registration(jsonpath)
    
def main_box():
    forbidden_boxes_file = "boxpos.txt"
    big_box_size = [512, 512, 343]  
    subbox_size = [64, 64, 32]  
    num_samples = 1000
    output_dir = "output"
    filename_prefix = "valid_positions"
    constraint_box = [13, 323, 120, 395, 64, 445]  # z_start, z_end, x_start, x_end, y_start, y_end

    # Charger les boîtes interdites
    forbidden_boxes = load_forbidden_boxes(forbidden_boxes_file)
    
    # Échantillonner et sauvegarder les sous-boîtes valides
    valid_positions = sample_subboxes(forbidden_boxes, big_box_size, subbox_size, num_samples, constraint_box)
    
    # Sauvegarder les positions valides
    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.join(output_dir, f"{filename_prefix}_positions.json")
    with open(json_filename, 'w') as f:
        json.dump(valid_positions, f)
    
    print(f"Les positions valides ont été sauvegardées dans : {json_filename}")
  
def resize_image(input_output):
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


def convert_tf_to_pytorch(for_training=False):
    # Load the TensorFlow model
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph('organs-5c-30fs-acc92-121.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.compat.v1.get_default_graph()
    x = graph.get_tensor_by_name("x_start:0")
    keepProb = graph.get_tensor_by_name("keepProb:0")
    feature_tensor = graph.get_tensor_by_name('MaxPool3D_1:0')

    # Save the model in SavedModel format
    tf.compat.v1.saved_model.simple_save(sess, "./saved_model_temp", 
                                         inputs={"x_start": x, "keepProb": keepProb},
                                         outputs={"MaxPool3D_1": feature_tensor})

    # Convert to ONNX using tf2onnx command-line tool
    cmd = f"python -m tf2onnx.convert --saved-model ./saved_model_temp --output tf_model_temp.onnx"
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print(result.stdout)

    # Convert ONNX to PyTorch
    pytorch_model = ConvertModel(onnx.load("tf_model_temp.onnx"))

    # Clean up temporary files
    os.remove("tf_model_temp.onnx")
    os.system("rm -rf ./saved_model_temp")

    if for_training:
        # Set requires_grad to True for all parameters
        for param in pytorch_model.parameters():
            param.requires_grad = True

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)

    return pytorch_model

def get_pytorch_model_for_inference():
    return convert_tf_to_pytorch(for_training=False)

def get_oscar_for_training():
    return convert_tf_to_pytorch(for_training=True)

if __name__ == "__main__":
    main_box()