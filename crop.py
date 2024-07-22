import SimpleITK as sitk
import numpy as np
import os

def pad_segmentation(seg_image, target_shape, start_index, end_index):
    print("Start pad_segmentation")
    seg_array = sitk.GetArrayFromImage(seg_image)
    padded_seg = np.zeros(target_shape, dtype=seg_array.dtype)
    print("padded_seg shape: ", padded_seg.shape)
    print("seg shape: ", seg_array.shape)
    print("start_index: ", start_index)
    print("end_index: ", end_index)
    padded_seg[start_index:end_index, :, :] = seg_array
    print("End pad_segmentation")
    return sitk.GetImageFromArray(padded_seg)

def find_closest_index(array, value):
    return np.argmin(np.abs(np.array(array) - value))

def crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder):
    print(f"Mask file: {mask_file}")
    print(f"Reference DICOM folder: {reference_dicom_folder}")
    
    # Lire le fichier de segmentation DICOM
    reader = sitk.ImageFileReader()
    reader.SetFileName(mask_file)
    reader.LoadPrivateTagsOn()
    seg_image = reader.Execute()

    # Lire les fichiers DICOM de référence
    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_names = dicom_series_reader.GetGDCMSeriesFileNames(reference_dicom_folder)
    dicom_series_reader.SetFileNames(dicom_names)
    reference_image = dicom_series_reader.Execute()

    # Obtenir les métadonnées nécessaires
    ref_size = reference_image.GetSize()
    ref_spacing = reference_image.GetSpacing()
    ref_direction = reference_image.GetDirection()
    ref_origin = reference_image.GetOrigin()

    # Calculer les indices de début et de fin pour le padding
    seg_size = seg_image.GetSize()
    start_index = 0
    end_index = seg_size[2]

    if ref_size[2] > seg_size[2]:
        start_index = (ref_size[2] - seg_size[2]) // 2
        end_index = start_index + seg_size[2]

    # Padding de l'image de segmentation
    padded_seg = pad_segmentation(seg_image, (ref_size[2], ref_size[1], ref_size[0]), start_index, end_index)
    padded_seg.SetSpacing(ref_spacing)
    padded_seg.SetDirection(ref_direction)
    padded_seg.SetOrigin(ref_origin)

    # Sauvegarder l'image complète paddée
    full_mask_path = os.path.splitext(output_path)[0] + "_full.nii.gz"
    sitk.WriteImage(padded_seg, full_mask_path)
    print(f"Full mask saved as {full_mask_path}")

    # Crop l'image
    crop_start = [crop_coords[4], crop_coords[2], crop_coords[0]]
    crop_size = [crop_coords[5] - crop_coords[4],
                 crop_coords[3] - crop_coords[2],
                 crop_coords[1] - crop_coords[0]]

    print(f"Image size: {padded_seg.GetSize()}")
    print(f"Crop start: {crop_start}")
    print(f"Crop size: {crop_size}")

    cropped_image = sitk.Crop(padded_seg, crop_start, crop_size)

    cropped_mask_path = os.path.splitext(output_path)[0] + "_cropped.nii.gz"
    sitk.WriteImage(cropped_image, cropped_mask_path)
    print(f"Cropped mask saved as {cropped_mask_path}")

    print(f"Cropped image size: {cropped_image.GetSize()}")

    # Écrire l'image croppée
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(cropped_image)

    print(f"Cropped image size: {cropped_image.GetSize()}")
    print(f"Mask cropped and saved as {output_path}")

def main():
    base_path = "/mnt/nas4/datasets/ToCurate/QA4IQI/FinalDataset-TCIA-MultiCentric/Upl/A1"
    reference_volume = "A1_174008_691000_SOMATOM_Definition_Edge_ID23_Harmonized_10mGy_IR_NrFiles_343"
    mask_file = os.path.join(base_path, reference_volume, "mask", f"{reference_volume}.dcm")
    reference_dicom_folder = os.path.join(base_path, reference_volume)

    # Coordonnées de crop [z_start, z_end, y_start, y_end, x_start, x_end]
    crop_coords = [13, 323, 120, 395, 64, 445]

    # Chemin de sortie pour le masque croppé
    output_path = os.path.join(base_path, f"{reference_volume}_cropped_mask.dcm")

    # Cropper le masque
    crop_volume(mask_file, output_path, crop_coords, reference_dicom_folder)

if __name__ == "__main__":
    main()