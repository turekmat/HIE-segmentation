import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_surface_distances(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0), sampling_ratio=0.5):
    """
    Optimized computation of surface distances.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (default: 1mm isotropic)
        sampling_ratio: Ratio of points used for computation (0-1), for speedup
    """
    pred_surface = np.logical_xor(pred_mask, binary_erosion(pred_mask))
    gt_surface = np.logical_xor(gt_mask, binary_erosion(gt_mask))

    gt_distance = distance_transform_edt(~gt_surface, sampling=spacing)

    if sampling_ratio < 1.0:
        pred_surface_points = np.argwhere(pred_surface)
        num_points = len(pred_surface_points)
        num_sampled = int(num_points * sampling_ratio)
        if num_sampled > 0:
            indices = np.random.choice(num_points, num_sampled, replace=False)
            surface_distances = gt_distance[tuple(pred_surface_points[indices].T)]
        else:
            surface_distances = gt_distance[pred_surface]
    else:
        surface_distances = gt_distance[pred_surface]

    return surface_distances

def compute_masd(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0), sampling_ratio=0.5):
    """
    Computation of Mean Average Surface Distance (MASD) between two masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (default: 1mm isotropic)
        sampling_ratio: Ratio of points used for computation (0-1), for speedup
        
    Returns:
        float: MASD value
    """
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 0.0

    distances_pred_to_gt = compute_surface_distances(pred_mask, gt_mask, spacing, sampling_ratio)
    distances_gt_to_pred = compute_surface_distances(gt_mask, pred_mask, spacing, sampling_ratio)

    if len(distances_pred_to_gt) == 0 and len(distances_gt_to_pred) == 0:
        return 0.0
    elif len(distances_pred_to_gt) == 0:
        return np.mean(distances_gt_to_pred)
    elif len(distances_gt_to_pred) == 0:
        return np.mean(distances_pred_to_gt)

    mean_distance = (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2.0
    return mean_distance

def compute_nsd(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0), tau=1.0, sampling_ratio=0.5):
    """
    Optimized computation of Normalized Surface Dice (NSD)
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (default: 1mm isotropic)
        tau: Distance tolerance (in mm)
        sampling_ratio: Ratio of points used for computation (0-1), for speedup
        
    Returns:
        float: NSD value
    """
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 1.0
    if not np.any(pred_mask) or not np.any(gt_mask):
        return 0.0

    distances_pred_to_gt = compute_surface_distances(
        pred_mask, gt_mask, spacing, sampling_ratio)
    distances_gt_to_pred = compute_surface_distances(
        gt_mask, pred_mask, spacing, sampling_ratio)

    pred_within_tau = np.sum(distances_pred_to_gt <= tau)
    gt_within_tau = np.sum(distances_gt_to_pred <= tau)

    total_pred_surface = len(distances_pred_to_gt) / sampling_ratio
    total_gt_surface = len(distances_gt_to_pred) / sampling_ratio

    nsd = (pred_within_tau + gt_within_tau) / (total_pred_surface + total_gt_surface)
    return nsd

def dice_coefficient(y_pred, y_true):
    """
    Computation of Dice coefficient between two binary masks.
    
    Args:
        y_pred: Predicted binary mask
        y_true: Ground truth binary mask
        
    Returns:
        float: Dice coefficient
    """
    epsilon = 1e-6
    y_pred = y_pred.astype(np.bool_)
    y_true = y_true.astype(np.bool_)
    
    pred_sum = np.sum(y_pred)
    true_sum = np.sum(y_true)
    if pred_sum == 0 and true_sum == 0:
        return 1.0
    
    intersection = np.sum(y_pred & y_true)
    return (2.0 * intersection) / (pred_sum + true_sum + epsilon) 