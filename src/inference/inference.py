import os
import torch
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

from ..utils.metrics import dice_coefficient, compute_masd, compute_nsd
from ..data.preprocessing import get_tta_transforms, apply_tta_transform, invert_tta_transform
from ..data.dataset import extract_patient_id

def tta_forward(model, input_tensor, device, tta_transforms):
    """
    Performs inference with Test-Time Augmentation for one input sample.

    Args:
        model: Trained model
        input_tensor: Torch tensor with shape (1, C, D, H, W)
        device: Target device ("cuda" or "cpu")
        tta_transforms: List of transformations (obtained using get_tta_transforms)

    Returns:
        np.ndarray: Average probability map (shape: (num_classes, D, H, W))
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=1)

    input_np = input_tensor.cpu().numpy()[0]  # shape: (C, D, H, W)
    accumulated_probs = None

    for transform in tta_transforms:
        aug_vol = apply_tta_transform(input_np, transform)
        aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
        
        with torch.no_grad():
            logits = model(aug_tensor)
        
        probs = softmax(logits)
        probs_np = probs.cpu().numpy()[0]
        inv_probs = invert_tta_transform(probs_np, transform)

        if accumulated_probs is None:
            accumulated_probs = inv_probs
        else:
            accumulated_probs += inv_probs

    avg_probs = accumulated_probs / len(tta_transforms)
    return avg_probs


def infer_full_volume(model, 
                     input_paths, 
                     label_path=None, 
                     device="cuda", 
                     use_tta=True, 
                     tta_angle_max=3,
                     training_mode="full_volume",
                     patch_size=(64, 64, 64),
                     batch_size=1,
                     use_z_adc=True):
    """
    Performs inference for a complete 3D volume.

    Args:
        model: Trained model
        input_paths: List of paths to input volumes (e.g., [adc_path, zadc_path])
        label_path: Path to ground truth mask (optional)
        device: Computation device
        use_tta: Whether to use Test-Time Augmentation
        tta_angle_max: Maximum angle for rotations during TTA
        training_mode: "full_volume" or "patch"
        patch_size: Patch size for patch-based inference
        batch_size: Batch size for patch-based inference
        use_z_adc: Whether to use Z-ADC modality (second input channel)

    Returns:
        dict: Inference results, including prediction and metrics (if ground truth is available)
    """
    model.eval()
    tta_transforms = get_tta_transforms(angle_max=tta_angle_max) if use_tta else None
    
    # Load input data
    volumes = []
    
    # Always load ADC map (first in the list)
    adc_path = input_paths[0]
    sitk_img = sitk.ReadImage(adc_path)
    np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    volumes.append(np_vol)
    
    # Load Z-ADC map, only if used
    if use_z_adc and len(input_paths) > 1:
        zadc_path = input_paths[1]
        try:
            sitk_zadc = sitk.ReadImage(zadc_path)
            zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
            volumes.append(zadc_np)
        except Exception as e:
            print(f"Warning: Cannot load Z-ADC file {zadc_path}: {e}")
            print("Inference will be performed with ADC map only.")
            use_z_adc = False  # Turn off Z-ADC if file cannot be loaded
    
    # Create input tensor
    input_vol = np.stack(volumes, axis=0)  # shape: (C, D, H, W)
    input_tensor = torch.from_numpy(input_vol).unsqueeze(0).to(device).float()
    
    # Ground truth data (if available)
    lab_np = None
    if label_path:
        lab_sitk = sitk.ReadImage(label_path)
        lab_np = sitk.GetArrayFromImage(lab_sitk)
    
    # Inference
    with torch.no_grad():
        if use_tta:
            avg_probs = 0
            for transform in tta_transforms:
                aug_vol = apply_tta_transform(input_vol, transform)
                aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
                
                if training_mode == "patch":
                    pred_logits = sliding_window_inference(
                        aug_tensor, patch_size, batch_size, model, overlap=0.25)
                else:
                    pred_logits = model(aug_tensor)
                
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(pred_logits)
                probs_np = probs.cpu().numpy()[0]
                inv_probs = invert_tta_transform(probs_np, transform)
                avg_probs += inv_probs
            
            avg_probs /= len(tta_transforms)
            pred_np = np.argmax(avg_probs, axis=0)
        else:
            if training_mode == "patch":
                pred_logits = sliding_window_inference(
                    input_tensor, patch_size, batch_size, model, overlap=0.25)
            else:
                pred_logits = model(input_tensor)
            
            pred_np = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
    
    # Calculate metrics (if ground truth is available)
    metrics = {}
    if lab_np is not None:
        metrics["dice"] = dice_coefficient(pred_np, lab_np)
        metrics["masd"] = compute_masd(pred_np, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
        metrics["nsd"] = compute_nsd(pred_np, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
        
    result = {
        'prediction': pred_np,
        'reference': lab_np,
        'input_paths': input_paths,
        'label_path': label_path,
        'metrics': metrics
    }
    
    if label_path:
        patient_id = extract_patient_id(label_path)
        result['patient_id'] = patient_id
    
    return result


def infer_full_volume_moe(main_model, expert_model, input_paths, label_path=None, 
                         device="cuda", threshold=80, use_z_adc=True):
    """
    Performs inference using a Mixture of Experts approach.

    Args:
        main_model: Main model
        expert_model: Expert model for small lesions
        input_paths: List of paths to input volumes
        label_path: Path to ground truth mask (optional)
        device: Computation device
        threshold: Threshold for switching to expert model
        use_z_adc: Whether to use Z-ADC modality (second input channel)

    Returns:
        dict: Inference results, including prediction and metrics
    """
    main_model.eval()
    expert_model.eval()
    
    # Load input data
    volumes = []
    
    # Always load ADC map (first in the list)
    adc_path = input_paths[0]
    sitk_img = sitk.ReadImage(adc_path)
    np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    volumes.append(np_vol)
    
    # Load Z-ADC map, only if used
    if use_z_adc and len(input_paths) > 1:
        zadc_path = input_paths[1]
        try:
            sitk_zadc = sitk.ReadImage(zadc_path)
            zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
            volumes.append(zadc_np)
        except Exception as e:
            print(f"Warning: Cannot load Z-ADC file {zadc_path}: {e}")
            print("Inference will be performed with ADC map only.")
            use_z_adc = False  # Turn off Z-ADC if file cannot be loaded
    
    # Create input tensor
    input_vol = np.stack(volumes, axis=0)  # shape: (C, D, H, W)
    input_tensor = torch.from_numpy(input_vol).unsqueeze(0).to(device).float()
    
    # Ground truth data (if available)
    lab_np = None
    if label_path:
        lab_sitk = sitk.ReadImage(label_path)
        lab_np = sitk.GetArrayFromImage(lab_sitk)
    
    # Inference with main model
    with torch.no_grad():
        logits_main = main_model(input_tensor)
        pred_main = torch.argmax(logits_main, dim=1).cpu().numpy()[0]
    
    # Determine number of foreground voxels
    fg_count = np.sum(pred_main == 1)
    
    # Select model based on foreground voxel count
    if fg_count < threshold and fg_count > 1:
        with torch.no_grad():
            logits_expert = expert_model(input_tensor)
            pred_final = torch.argmax(logits_expert, dim=1).cpu().numpy()[0]
        model_used = 'expert'
    else:
        pred_final = pred_main
        model_used = 'main'
    
    # Calculate metrics (if ground truth is available)
    metrics = {}
    if lab_np is not None:
        metrics["dice"] = dice_coefficient(pred_final, lab_np)
        metrics["masd"] = compute_masd(pred_final, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
        metrics["nsd"] = compute_nsd(pred_final, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
    
    result = {
        'prediction': pred_final,
        'reference': lab_np,
        'input_paths': input_paths,
        'label_path': label_path,
        'metrics': metrics,
        'model_used': model_used,
        'foreground_voxels': fg_count
    }
    
    if label_path:
        patient_id = extract_patient_id(label_path)
        result['patient_id'] = patient_id
    
    return result


def save_segmentation_to_file(prediction, reference_sitk, output_path):
    """
    Saves the predicted segmentation to a file.

    Args:
        prediction: Predicted segmentation (numpy array)
        reference_sitk: Reference SimpleITK image for metadata
        output_path: Path to save the segmentation
    """
    # Create SimpleITK image from numpy array
    prediction = prediction.astype(np.uint8)
    output_sitk = sitk.GetImageFromArray(prediction)
    
    # Copy metadata from reference image
    output_sitk.SetSpacing(reference_sitk.GetSpacing())
    output_sitk.SetOrigin(reference_sitk.GetOrigin())
    output_sitk.SetDirection(reference_sitk.GetDirection())
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    sitk.WriteImage(output_sitk, output_path)
    print(f"Segmentation saved to: {output_path}")


def save_slice_comparison_pdf(result, output_dir, prefix="comparison"):
    """
    Creates a PDF file comparing all slices of ground truth and model prediction.
    The PDF will contain two columns - ground truth in one column and prediction in the other.
    
    Args:
        result: Result from infer_full_volume or infer_full_volume_moe
        output_dir: Output directory
        prefix: Prefix for filename
        
    Returns:
        str: Path to the created PDF file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction = result['prediction']  # shape (D, H, W)
    reference = result['reference']    # shape (D, H, W)
    
    if reference is None:
        print("Ground truth reference is not available, cannot create PDF comparison.")
        return None
    
    # Get metadata for filename
    metrics = result.get('metrics', {})
    dice = metrics.get('dice', 0.0)
    patient_id = result.get('patient_id', 'unknown')
    
    # Create file path
    output_name = f"{prefix}_{patient_id}_dice{dice:.3f}.pdf"
    output_path = os.path.join(output_dir, output_name)
    
    # Determine number of slices and define colors for visualization
    num_slices = reference.shape[0]
    rows_per_page = 6  # Number of slices per page
    
    # Define colormaps for segmentations (transparent background, red lesion)
    gt_cmap = plt.cm.colors.ListedColormap(['none', 'red'])
    pred_cmap = plt.cm.colors.ListedColormap(['none', 'blue'])
    
    # Load input image for background, if available
    background_vol = None
    input_path = result['input_paths'][0]
    if os.path.exists(input_path):
        try:
            adc_sitk = sitk.ReadImage(input_path)
            background_vol = sitk.GetArrayFromImage(adc_sitk)
            # Normalize for display
            background_vol = (background_vol - background_vol.min()) / (background_vol.max() - background_vol.min() + 1e-8)
        except Exception as e:
            print(f"Cannot load background from ADC: {e}")
            background_vol = None
    
    # Create PDF with multiple slices per page
    with PdfPages(output_path) as pdf:
        num_pages = math.ceil(num_slices / rows_per_page)
        
        for page in range(num_pages):
            start_idx = page * rows_per_page
            end_idx = min(start_idx + rows_per_page, num_slices)
            slices_on_page = end_idx - start_idx
            
            # Create grid for current page
            fig, axes = plt.subplots(slices_on_page, 2, figsize=(10, 2 * slices_on_page))
            
            # Ensure axes is always a 2D array, even if there's only one slice
            if slices_on_page == 1:
                axes = np.array([axes])
            
            # Process slices for current page
            for i in range(slices_on_page):
                slice_idx = start_idx + i
                row = i
                
                # Reference slice (Ground Truth)
                if background_vol is not None:
                    # Show ADC in background and segmentation over it
                    bg_slice = background_vol[slice_idx, :, :]
                    axes[row, 0].imshow(bg_slice, cmap='gray')
                    axes[row, 0].imshow(reference[slice_idx, :, :], cmap=gt_cmap, alpha=0.7)
                else:
                    # Show segmentation only
                    axes[row, 0].imshow(reference[slice_idx, :, :], cmap='gray')
                
                axes[row, 0].set_title(f'Ground Truth (slice {slice_idx+1})')
                axes[row, 0].axis('off')
                
                # Predicted slice (Prediction)
                if background_vol is not None:
                    # Show ADC in background and segmentation over it
                    bg_slice = background_vol[slice_idx, :, :]
                    axes[row, 1].imshow(bg_slice, cmap='gray')
                    axes[row, 1].imshow(prediction[slice_idx, :, :], cmap=pred_cmap, alpha=0.7)
                else:
                    # Show segmentation only
                    axes[row, 1].imshow(prediction[slice_idx, :, :], cmap='gray')
                
                axes[row, 1].set_title(f'Prediction (slice {slice_idx+1})')
                axes[row, 1].axis('off')
            
            # Add main title to page
            plt.suptitle(f'Segmentation comparison - patient {patient_id}, Dice coefficient: {dice:.3f}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reserve space for main title
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"PDF with slice comparison saved to: {output_path}")
    return output_path


def save_validation_results_pdf(result, output_dir, prefix="validation"):
    """
    Creates a PDF file with all slices of validation sample with three separate columns: 
    ZADC map, LABEL map and model prediction.
    
    Args:
        result: Result from infer_full_volume or infer_full_volume_moe
        output_dir: Output directory
        prefix: Prefix for filename
        
    Returns:
        str: Path to the created PDF file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction = result['prediction']  # shape (D, H, W)
    reference = result['reference']    # shape (D, H, W)
    
    if reference is None:
        print("Ground truth reference is not available, cannot create PDF comparison.")
        return None
    
    # Get metadata for filename
    metrics = result.get('metrics', {})
    dice = metrics.get('dice', 0.0)
    masd = metrics.get('masd', 0.0)
    nsd = metrics.get('nsd', 0.0)
    patient_id = result.get('patient_id', 'unknown')
    
    # Create file path
    output_name = f"{prefix}_{patient_id}_dice{dice:.3f}.pdf"
    output_path = os.path.join(output_dir, output_name)
    
    # Determine number of slices and define colors for visualization
    num_slices = reference.shape[0]
    rows_per_page = 6  # Number of slices per page
    
    # Define colormaps for segmentations
    label_cmap = 'jet'  # More colors for label
    pred_cmap = 'jet'   # More colors for prediction
    
    # Load input images
    zadc_vol = None
    if len(result['input_paths']) > 1:
        try:
            zadc_path = result['input_paths'][1]  # ZADC is second input
            zadc_sitk = sitk.ReadImage(zadc_path)
            zadc_vol = sitk.GetArrayFromImage(zadc_sitk)
            # Normalize for display
            if np.max(zadc_vol) != np.min(zadc_vol):
                zadc_vol = (zadc_vol - np.min(zadc_vol)) / (np.max(zadc_vol) - np.min(zadc_vol))
        except Exception as e:
            print(f"Cannot load ZADC map: {e}")
            zadc_vol = None
    
    if zadc_vol is None:
        print("ZADC map is not available, using ADC map instead of ZADC.")
        try:
            adc_path = result['input_paths'][0]
            adc_sitk = sitk.ReadImage(adc_path)
            zadc_vol = sitk.GetArrayFromImage(adc_sitk)
            # Normalize for display
            if np.max(zadc_vol) != np.min(zadc_vol):
                zadc_vol = (zadc_vol - np.min(zadc_vol)) / (np.max(zadc_vol) - np.min(zadc_vol))
        except Exception as e:
            print(f"Cannot load ADC map either: {e}")
            return None
    
    # Create PDF with multiple slices per page
    with PdfPages(output_path) as pdf:
        num_pages = math.ceil(num_slices / rows_per_page)
        
        for page in range(num_pages):
            start_idx = page * rows_per_page
            end_idx = min(start_idx + rows_per_page, num_slices)
            slices_on_page = end_idx - start_idx
            
            # Create grid for current page - 3 columns: ZADC, LABEL, PRED
            fig, axes = plt.subplots(slices_on_page, 3, figsize=(15, 2 * slices_on_page))
            
            # Ensure axes is always a 2D array, even if there's only one slice
            if slices_on_page == 1:
                axes = np.array([axes])
            
            # Process slices for current page
            for i in range(slices_on_page):
                slice_idx = start_idx + i
                row = i
                
                # ZADC map
                zadc_slice = zadc_vol[slice_idx, :, :]
                axes[row, 0].imshow(zadc_slice, cmap='gray')
                
                # For first slice (slice_idx == 0) add information about all metrics
                if slice_idx == 0:
                    axes[row, 0].set_title(f'ZADC (slice 1) - DICE: {dice:.4f}')
                    axes[row, 1].set_title(f'LABEL (slice 1) - MASD: {masd:.4f}')
                    axes[row, 2].set_title(f'PRED (slice 1) - NSD: {nsd:.4f}')
                else:
                    axes[row, 0].set_title(f'ZADC (slice {slice_idx+1})')
                    axes[row, 1].set_title(f'LABEL (slice {slice_idx+1})')
                    axes[row, 2].set_title(f'PRED (slice {slice_idx+1})')
                
                axes[row, 0].axis('off')
                
                # LABEL map (Ground Truth)
                axes[row, 1].imshow(reference[slice_idx, :, :], cmap=label_cmap)
                axes[row, 1].axis('off')
                
                # Model prediction
                axes[row, 2].imshow(prediction[slice_idx, :, :], cmap=pred_cmap)
                axes[row, 2].axis('off')
            
            # Add main title to page
            if page == 0:
                # On first page show all metrics in main title
                plt.suptitle(f'Patient {patient_id} - Metrics: DICE: {dice:.4f}, MASD: {masd:.4f}, NSD: {nsd:.4f}', 
                             fontsize=14)
            else:
                plt.suptitle(f'Validation visualization - patient {patient_id}', fontsize=14)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reserve space for main title
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"PDF with validation results saved to: {output_path}")
    return output_path


def save_segmentation_with_metrics(result, output_dir, prefix="segmentation", save_pdf_comparison=False):
    """
    Saves segmentation and adds metric values to the filename.

    Args:
        result: Result from infer_full_volume or infer_full_volume_moe
        output_dir: Output directory
        prefix: Prefix for filename
        save_pdf_comparison: Whether to create PDF with slice comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction = result['prediction']
    input_path = result['input_paths'][0]  # First input path for reference
    ref_sitk = sitk.ReadImage(input_path)
    
    metrics = result.get('metrics', {})
    dice = metrics.get('dice', 0.0)
    masd = metrics.get('masd', 0.0)
    nsd = metrics.get('nsd', 0.0)
    
    patient_id = result.get('patient_id', 'unknown')
    output_name = f"{prefix}_{patient_id}_dice{dice:.3f}_masd{masd:.3f}_nsd{nsd:.3f}.mha"
    output_path = os.path.join(output_dir, output_name)
    
    save_segmentation_to_file(prediction, ref_sitk, output_path)
    
    # If requested, create PDF comparison
    if save_pdf_comparison and result.get('reference') is not None:
        save_slice_comparison_pdf(result, output_dir, prefix=f"{prefix}_comparison")
    
    return output_path