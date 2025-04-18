import os
import numpy as np
import SimpleITK as sitk
import random
import re
from scipy.ndimage import rotate, gaussian_filter, distance_transform_edt, binary_erosion
import torch

def filter_augmented_files(file_list, max_aug):
    """
    Filters augmented files based on the original file.
    Keeps original files and limits the number of augmented
    files to max_aug per original file.

    Args:
        file_list: List of files to filter
        max_aug: Maximum number of augmented files per original

    Returns:
        Filtered list of files
    """
    grouped = {}
    for f in file_list:
        key = get_base_id(f)
        if '_aug' in f.lower():

            if key not in grouped:
                grouped[key] = {'orig': None, 'aug': []}
            grouped[key]['aug'].append(f)
        else:

            if key not in grouped:
                grouped[key] = {'orig': None, 'aug': []}
            grouped[key]['orig'] = f

    filtered_list = []
    for key in sorted(grouped.keys()):
        entry = grouped[key]
        if entry['orig'] is not None:
            filtered_list.append(entry['orig'])
            selected_aug = sorted(entry['aug'])[:max_aug]
            filtered_list.extend(selected_aug)
        else:
            selected = sorted(entry['aug'])[:max_aug]
            filtered_list.extend(selected)
    return filtered_list


def get_base_id(filename: str):
    """
    Gets the base ID of the file without the suffix _aug.
    """
    filename_lower = filename.lower()
    if '_aug' in filename_lower:
        return re.sub(r'_aug\d+.*', '', filename_lower)
    else:
        return filename_lower


def soft_3d_augmentation(
    adc_np,
    zadc_np,
    label_np,
    angle_max=3,
    p_flip=0.5,
    p_noise=0.0,
    noise_std=0.01,
    p_smooth=0.0,
    smooth_sigma=1.0
):
    """
    Performs a light 3D volume augmentation.

    Args:
        adc_np: Numpy array ADC volume
        zadc_np: Numpy array Z-ADC volume
        label_np: Numpy array mask
        angle_max: Maximum rotation angle
        p_flip: Probability of horizontal flip
        p_noise: Probability of adding noise
        noise_std: Standard deviation for noise
        p_smooth: Probability of smoothing
        smooth_sigma: Sigma for Gaussian smoothing

    Returns:
        Augmented volumes (adc_np, zadc_np, label_np)
    """
    if random.random() < p_flip:
      adc_np = np.flip(adc_np, axis=2).copy()
      zadc_np = np.flip(zadc_np, axis=2).copy()
      label_np = np.flip(label_np, axis=2).copy()

    angle = random.uniform(-angle_max, angle_max)
    axes  = random.choice([(0, 1), (0, 2), (1, 2)])  # choose one pair of axes
    adc_np = rotate(adc_np,   angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    zadc_np = rotate(zadc_np, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    label_np = rotate(label_np, angle=angle, axes=axes, reshape=False, order=0, mode='nearest')

    if random.random() < p_noise:
        noise_adc = np.random.normal(0, noise_std, size=adc_np.shape)
        noise_z   = np.random.normal(0, noise_std, size=zadc_np.shape)
        adc_np   = adc_np + noise_adc
        zadc_np  = zadc_np + noise_z

    if random.random() < p_smooth:
        adc_np   = gaussian_filter(adc_np, sigma=smooth_sigma)
        zadc_np  = gaussian_filter(zadc_np, sigma=smooth_sigma)

    return adc_np, zadc_np, label_np


def heavy_3d_augmentation(
    adc_np,
    zadc_np,
    label_np,
    angle_max=15,
    p_flip_x=0.5,
    p_flip_y=0.5,
    p_flip_z=0.5,
    p_noise=0.5,
    noise_std_adc=0.02,
    noise_std_zadc=0.2,
    p_smooth=0.3,
    smooth_sigma_range=(0.5, 2.0),
    p_intensity=0.7
):
    """
    Performs a stronger 3D volume augmentation, suitable for increasing model robustness.

    Args:
        adc_np: Numpy array ADC volume (range 0-1)
        zadc_np: Numpy array Z-ADC volume (range -10 to +10)
        label_np: Numpy array mask
        angle_max: Maximum rotation angle (±degrees)
        p_flip_x: Probability of flip along X axis
        p_flip_y: Probability of flip along Y axis
        p_flip_z: Probability of flip along Z axis
        p_noise: Probability of adding noise
        noise_std_adc: Standard deviation for noise in ADC map
        noise_std_zadc: Standard deviation for noise in Z-ADC map
        p_smooth: Probability of smoothing
        smooth_sigma_range: Range of sigma parameter for Gaussian smoothing
        p_intensity: Probability of intensity adjustment

    Returns:
        Augmented volumes (adc_np, zadc_np, label_np)
    """

    if random.random() < p_flip_x:
        adc_np = np.flip(adc_np, axis=0).copy()
        zadc_np = np.flip(zadc_np, axis=0).copy()
        label_np = np.flip(label_np, axis=0).copy()
    
    if random.random() < p_flip_y:
        adc_np = np.flip(adc_np, axis=1).copy()
        zadc_np = np.flip(zadc_np, axis=1).copy()
        label_np = np.flip(label_np, axis=1).copy()
    
    if random.random() < p_flip_z:
        adc_np = np.flip(adc_np, axis=2).copy()
        zadc_np = np.flip(zadc_np, axis=2).copy()
        label_np = np.flip(label_np, axis=2).copy()

    angle = random.uniform(-angle_max, angle_max)
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    adc_np = rotate(adc_np, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    zadc_np = rotate(zadc_np, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    label_np = rotate(label_np, angle=angle, axes=axes, reshape=False, order=0, mode='nearest')


    if random.random() < p_intensity:

        brightness_factor_adc = random.uniform(-0.1, 0.1)
        

        contrast_factor_adc = random.uniform(0.8, 1.2)
        
        gamma_factor_adc = random.uniform(0.8, 1.2)
        

        mask_nonzero = adc_np > 0
        

        adc_np[mask_nonzero] = adc_np[mask_nonzero] + brightness_factor_adc
        
        mean_adc = np.mean(adc_np[mask_nonzero])
        adc_np[mask_nonzero] = (adc_np[mask_nonzero] - mean_adc) * contrast_factor_adc + mean_adc
        

        adc_np_positive = np.maximum(adc_np[mask_nonzero], 1e-8)
        adc_np[mask_nonzero] = np.power(adc_np_positive, gamma_factor_adc)
        
        adc_np = np.clip(adc_np, 0, 1)
        
        brightness_factor_zadc = random.uniform(-1.0, 1.0)
        
        contrast_factor_zadc = random.uniform(0.85, 1.15)
        
        mask_nonzero_z = zadc_np != 0
        
        zadc_np[mask_nonzero_z] = zadc_np[mask_nonzero_z] + brightness_factor_zadc
        
        mean_zadc = np.mean(zadc_np[mask_nonzero_z]) if np.any(mask_nonzero_z) else 0
        zadc_np[mask_nonzero_z] = (zadc_np[mask_nonzero_z] - mean_zadc) * contrast_factor_zadc + mean_zadc

        zadc_np = np.clip(zadc_np, -15, 15)

    # 4. Gaussian noise (different parameters for ADC and ZADC)
    if random.random() < p_noise:
        # For ADC (0-1)
        noise_adc = np.random.normal(0, noise_std_adc, size=adc_np.shape)
        adc_np = adc_np + noise_adc
        adc_np = np.clip(adc_np, 0, 1)
        
        # For ZADC (-10 to +10)
        noise_zadc = np.random.normal(0, noise_std_zadc, size=zadc_np.shape)
        zadc_np = zadc_np + noise_zadc
        zadc_np = np.clip(zadc_np, -15, 15)

    # 5. Gaussian smoothing with variable parameter
    if random.random() < p_smooth:
        sigma = random.uniform(smooth_sigma_range[0], smooth_sigma_range[1])
        adc_np = gaussian_filter(adc_np, sigma=sigma)
        zadc_np = gaussian_filter(zadc_np, sigma=sigma)

    return adc_np, zadc_np, label_np


def prepare_preprocessed_data(
    adc_folder, z_folder, label_folder,
    output_adc, output_z, output_label,
    normalize=True,
    allow_normalize_spacing=False
):
    """
    Preprocesses data from original folders into output folders.
    Implementation based on optimized preprocessing.
    
    Args:
        adc_folder: Input folder with ADC scans
        z_folder: Input folder with Z-ADC scans
        label_folder: Input folder with masks
        output_adc: Output folder for preprocessed ADC scans
        output_z: Output folder for preprocessed Z-ADC scans
        output_label: Output folder for preprocessed masks
        normalize: Whether to normalize intensities (True/False)
        allow_normalize_spacing: Whether to normalize spacing (True/False)
    """
    # Check and create output directories
    for output_dir in [output_adc, output_z, output_label]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Check if preprocessed files already exist
    existing_files = [
        len([f for f in os.listdir(dir) if f.endswith('.mha')]) 
        for dir in [output_adc, output_z, output_label]
    ]
    
    if all(count > 0 for count in existing_files) and len(set(existing_files)) == 1:
        print("Preprocessed data already exist. Skipping preprocessing.")
        return
    
    # Load file list
    adc_files = sorted([f for f in os.listdir(adc_folder) if f.endswith('.mha')])
    z_files = sorted([f for f in os.listdir(z_folder) if f.endswith('.mha')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mha')])
    
    if not (len(adc_files) == len(z_files) == len(label_files)):
        raise ValueError("Inconsistent number of input files across folders.")
    
    print(f"Processing {len(adc_files)} file sets...")
    
    # Process each file
    for i, (adc_name, zadc_name, label_name) in enumerate(zip(adc_files, z_files, label_files)):
        print(f"Processing {i+1}/{len(adc_files)}: {adc_name}")
        
        # Load input scans
        adc_img = sitk.ReadImage(os.path.join(adc_folder, adc_name))
        zadc_img = sitk.ReadImage(os.path.join(z_folder, zadc_name))
        label_img = sitk.ReadImage(os.path.join(label_folder, label_name))
        
        # 1. Normalize spacing - as first step if allowed
        if allow_normalize_spacing:
            adc_img = normalize_spacing(adc_img)
            zadc_img = normalize_spacing(zadc_img)
            label_img = normalize_spacing(label_img)
        
        # Save metadata for later use
        final_spacing = adc_img.GetSpacing()
        final_direction = adc_img.GetDirection()
        final_origin = adc_img.GetOrigin()
        
        # 2. Convert to numpy arrays
        adc_np = sitk.GetArrayFromImage(adc_img)
        zadc_np = sitk.GetArrayFromImage(zadc_img)
        label_np = sitk.GetArrayFromImage(label_img)
        
        # 3. Create brain mask and normalize
        if normalize:
            brain_mask = (adc_np > 0) & (zadc_np > 0)
            adc_np = z_score_normalize(adc_np, brain_mask)
            zadc_np = z_score_normalize(zadc_np, brain_mask)
        
        # 4. Compute and apply bounding box
        bounding_box = compute_largest_3d_bounding_box([adc_np, zadc_np], threshold=0)
        adc_np, zadc_np, label_np = crop_to_largest_bounding_box(adc_np, zadc_np, label_np, bounding_box, margin=5)
        
        # 5. Padding to multiples of 32
        adc_np = pad_3d_all_dims_to_multiple_of_32(adc_np)
        zadc_np = pad_3d_all_dims_to_multiple_of_32(zadc_np)
        label_np = pad_3d_all_dims_to_multiple_of_32(label_np)
        
        # Convert back to SimpleITK images
        processed_adc = sitk.GetImageFromArray(adc_np)
        processed_z = sitk.GetImageFromArray(zadc_np)
        processed_label = sitk.GetImageFromArray(label_np)
        
        # Set metadata
        for img in [processed_adc, processed_z, processed_label]:
            img.SetSpacing(final_spacing)
            img.SetDirection(final_direction)
            img.SetOrigin(final_origin)
        
        # Save preprocessed files
        sitk.WriteImage(processed_adc, os.path.join(output_adc, adc_name))
        sitk.WriteImage(processed_z, os.path.join(output_z, zadc_name))
        sitk.WriteImage(processed_label, os.path.join(output_label, label_name))
        
        # Free memory
        del adc_np, zadc_np, label_np, processed_adc, processed_z, processed_label
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Preprocessing complete.")


def compute_largest_3d_bounding_box(volumes, threshold=0):
    """
    Computes the largest bounding box from all input volumes.
    """
    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])

    for volume in volumes:
        nonzero = np.argwhere(volume > threshold)
        if nonzero.size > 0:
            min_coords = np.minimum(min_coords, nonzero.min(axis=0))
            max_coords = np.maximum(max_coords, nonzero.max(axis=0) + 1)

    return ((int(min_coords[0]), int(max_coords[0])),
            (int(min_coords[1]), int(max_coords[1])),
            (int(min_coords[2]), int(max_coords[2])))


def crop_to_largest_bounding_box(adc_np, zadc_np, label_np, bounding_box, margin=5):
    """
    Crops volumes based on the bounding box with added margin.
    """
    (minD, maxD), (minH, maxH), (minW, maxW) = bounding_box

    # Add margin
    minD = max(0, minD - margin)
    maxD = min(adc_np.shape[0], maxD + margin)
    minH = max(0, minH - margin)
    maxH = min(adc_np.shape[1], maxH + margin)
    minW = max(0, minW - margin)
    maxW = min(adc_np.shape[2], maxW + margin)

    # Crop volumes
    return (adc_np[minD:maxD, minH:maxH, minW:maxW],
            zadc_np[minD:maxD, minH:maxH, minW:maxW],
            label_np[minD:maxD, minH:maxH, minW:maxW])


def pad_3d_all_dims_to_multiple_of_32(volume_3d, mode="edge"):
    """
    Adds padding to a 3D volume so that all dimensions are multiples of 32.
    """
    def pad_dim_to_32(dim_size):
        return ((dim_size - 1) // 32 + 1) * 32 if dim_size % 32 != 0 else dim_size

    newD, newH, newW = [pad_dim_to_32(dim) for dim in volume_3d.shape]
    padD, padH, padW = newD - volume_3d.shape[0], newH - volume_3d.shape[1], newW - volume_3d.shape[2]

    return np.pad(
        volume_3d,
        pad_width=((0, padD), (0, padH), (0, padW)),
        mode=mode
    )


def normalize_spacing(image_sitk, target_spacing=(1.0, 1.0, 1.0)):
    """
    Normalizes the spacing of the image to the target spacing using trilinear interpolation
    for image data and nearest neighbor for mask data.
    
    This ensures uniformity in image resolution by interpolating all images to a fixed resolution
    of 1mm × 1mm × 1mm, preserving the spatial meaning of data.
    
    Args:
        image_sitk: SimpleITK image to normalize
        target_spacing: Target spacing (default: 1.0, 1.0, 1.0) in mm
        
    Returns:
        Normalized SimpleITK image with target spacing
    """
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    new_size = [int(round(osz * ospacing / tspacing))
                for osz, ospacing, tspacing in zip(original_size, original_spacing, target_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    # Interpolation: trilinear for images, nearest neighbor for masks
    if image_sitk.GetPixelID() in [sitk.sitkUInt8, sitk.sitkInt8]:  # probably mask
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:  # probably image - use trilinear interpolation
        resample.SetInterpolator(sitk.sitkLinear)  # trilinear interpolation

    return resample.Execute(image_sitk)


def z_score_normalize(image_np, mask=None):
    """
    Performs Z-score normalization of the volume (subtracting mean and dividing by standard deviation).
    If a mask is provided, normalization is performed only on voxels in the mask.
    """
    valid_voxels = image_np[mask > 0] if mask is not None else image_np
    mean, std = np.mean(valid_voxels), np.std(valid_voxels)
    return (image_np - mean) / (std if std != 0 else 1.0)


def apply_tta_transform(volume, transform):
    """
    Applies transformation for Test-Time Augmentation.
    """
    do_flip = transform.get('flip', False)
    rotation = transform.get('rotation', None)
    
    # If we have (C, D, H, W), we work with each channel separately
    if len(volume.shape) == 4:
        C, D, H, W = volume.shape
        transformed = np.zeros_like(volume)
        for c in range(C):
            transformed[c] = apply_tta_transform(volume[c], transform)
        return transformed
    
    # For standalone volume (D, H, W)
    result = volume.copy()
    
    # Horizontal flip (if requested)
    if do_flip:
        result = np.flip(result, axis=2)
    
    # Rotation (if requested)
    if rotation is not None:
        angle = rotation['angle']
        axes = rotation['axes']
        result = rotate(result, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    
    return result


def invert_tta_transform(volume, transform):
    """
    Inverts the TTA transformation so that results can be averaged.
    """
    do_flip = transform.get('flip', False)
    rotation = transform.get('rotation', None)
    
    if len(volume.shape) == 4:
        C, D, H, W = volume.shape
        inverted = np.zeros_like(volume)
        for c in range(C):
            inverted[c] = invert_tta_transform(volume[c], transform)
        return inverted
    
    result = volume.copy()
    
    if rotation is not None:
        angle = -rotation['angle']
        axes = rotation['axes']
        result = rotate(result, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    
    if do_flip:
        result = np.flip(result, axis=2)
    
    return result


def get_tta_transforms(angle_max=3):
    """
    Creates a list of transformations for Test-Time Augmentation.
    """
    tta_transforms = []
    flips = [False, True]
    rotations = [None,
                 {"angle": angle_max,  "axes": (0, 1)},
                 {"angle": -angle_max, "axes": (0, 1)},
                 {"angle": angle_max,  "axes": (0, 2)},
                 {"angle": -angle_max, "axes": (0, 2)},
                 {"angle": angle_max,  "axes": (1, 2)},
                 {"angle": -angle_max, "axes": (1, 2)}]

    for flip in flips:
        for rot in rotations:
            tta_transforms.append({"flip": flip, "rotation": rot})
    return tta_transforms


def select_inpainted_data_for_training(original_adc_path, original_z_path, original_label_path, 
                                       inpaint_adc_dir, inpaint_z_dir, inpaint_label_dir,
                                       probability=0.2):
    """
    Selects inpainted data for training with a certain probability instead of original data.
    
    Args:
        original_adc_path: Path to the original ADC file
        original_z_path: Path to the original Z-ADC file
        original_label_path: Path to the original LABEL file
        inpaint_adc_dir: Directory with inpainted ADC files
        inpaint_z_dir: Directory with inpainted Z-ADC files
        inpaint_label_dir: Directory with inpainted LABEL files
        probability: Probability of selecting inpainted data (0.0-1.0)
        
    Returns:
        Tuple (adc_path, z_path, label_path) with paths to files that should be used for training
    """
    if random.random() > probability:
        return original_adc_path, original_z_path, original_label_path
    
    original_adc_basename = os.path.basename(original_adc_path)
    match = re.match(r'(.*?-VISIT_\d+)', original_adc_basename)
    if not match:
        print(f"Warning: Could not extract prefix from {original_adc_basename}")
        return original_adc_path, original_z_path, original_label_path
    
    prefix = match.group(1)
    
    inpaint_adc_files = [f for f in os.listdir(inpaint_adc_dir) 
                        if f.startswith(prefix) and 'LESIONED_ADC' in f]
    
    if not inpaint_adc_files:
        print(f"Info: No inpainted files found for {prefix}")
        return original_adc_path, original_z_path, original_label_path
    
    selected_inpaint_adc = random.choice(inpaint_adc_files)
    
    match = re.search(r'sample(\d+)', selected_inpaint_adc)
    if not match:
        return original_adc_path, original_z_path, original_label_path
    
    sample_id = match.group(1)
    
    inpaint_z_pattern = f"{prefix}-.*sample{sample_id}-LESIONED_ZADC.mha"
    inpaint_label_pattern = f"{prefix}-.*sample{sample_id}_combined_lesion.mha"
    
    inpaint_z_matches = [f for f in os.listdir(inpaint_z_dir) 
                         if re.match(inpaint_z_pattern, f)]
    inpaint_label_matches = [f for f in os.listdir(inpaint_label_dir) 
                            if re.match(inpaint_label_pattern, f)]
    
    if not inpaint_z_matches or not inpaint_label_matches:
        print(f"Warning: Couldn't find complete inpainted set for {prefix} sample{sample_id}")
        return original_adc_path, original_z_path, original_label_path
    
    inpaint_adc_path = os.path.join(inpaint_adc_dir, selected_inpaint_adc)
    inpaint_z_path = os.path.join(inpaint_z_dir, inpaint_z_matches[0])
    inpaint_label_path = os.path.join(inpaint_label_dir, inpaint_label_matches[0])
    
    print(f"Using inpainted data: {prefix} (sample{sample_id})")
    
    return inpaint_adc_path, inpaint_z_path, inpaint_label_path

# Backward compatibility - alias function
random_3d_augmentation = soft_3d_augmentation 