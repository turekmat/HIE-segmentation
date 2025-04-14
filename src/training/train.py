import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import wandb
from monai.inferers import sliding_window_inference

from ..models import create_model
from ..loss import dice_coefficient
from ..data.preprocessing import get_tta_transforms
from ..data.dataset import get_subject_id_from_filename


def custom_collate_fn(batch):
    """
    Custom collate function that allows processing tensors of different sizes.
    Each sample is processed individually, without stack operation.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        List containing pairs of (inputs, labels)
    """
    return batch


def train_one_epoch(model, loader, optimizer, loss_fn, device="cuda"):
    """
    Trains the model for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader with training data
        optimizer: Optimizer
        loss_fn: Loss function
        device: Computation device
        
    Returns:
        float: Average loss for the entire epoch
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def optimal_sliding_window_inference(model, inputs, roi_size, sw_batch_size, overlap=0.5, mode="gaussian", device=None):
    """
    Enhanced version of sliding window inference with advanced prediction weighting.
    
    Args:
        model: Model for inference
        inputs: Input tensor (B, C, D, H, W)
        roi_size: Patch size for inference (D, H, W)
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio between adjacent patches (0-1)
        mode: Method for weighting overlapping predictions ("gaussian" or "constant")
        device: Computation device
        
    Returns:
        torch.Tensor: Prediction for the entire input
    """
    if device is None:
        device = inputs.device
    
    sigma_scale = 0.125
    extra_params = {}
    
    if mode == "gaussian":
        extra_params = {
            "mode": "gaussian", 
            "sigma_scale": sigma_scale,
            "padding_mode": "reflect"
        }
    else:
        extra_params = {"mode": "constant"}
    
    input_shape = list(inputs.shape[2:])  # [D, H, W]
    patch_shape = roi_size
    
    adaptive_overlap = [overlap] * len(input_shape)
    for i in range(len(input_shape)):
        if patch_shape[i] >= input_shape[i]:
            adaptive_overlap[i] = 0.01
            patch_shape[i] = min(patch_shape[i], input_shape[i])
    
    try:
        return sliding_window_inference(
            inputs=inputs,
            roi_size=patch_shape,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=adaptive_overlap,
            **extra_params
        )
    except RuntimeError as e:
        print(f"Warning: Error during sliding window inference. Trying with smaller overlap. Original error: {e}")
        try:
            return sliding_window_inference(
                inputs=inputs,
                roi_size=patch_shape,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.01,  # Minimal overlap
                **extra_params
            )
        except Exception as e2:
            print(f"Warning: Fallback solution failed. Attempting to process the entire input at once. Error: {e2}")
            if inputs.shape[2] * inputs.shape[3] * inputs.shape[4] > 128 * 128 * 64:
                print("Warning: Input is very large for processing at once!")
            return model(inputs)


def validate_one_epoch(model, loader, loss_fn, device="cuda", training_mode="full_volume",
                       compute_surface_metrics=True, USE_TTA=True, TTA_ANGLE_MAX=3,
                       batch_size=1, patch_size=(64, 64, 64), tta_forward_fn=None,
                       sw_overlap=0.5):
    """
    Validates the model on the validation set.
    
    Args:
        model: Model to validate
        loader: DataLoader with validation data
        loss_fn: Loss function
        device: Computation device
        training_mode: "full_volume" or "patch"
        compute_surface_metrics: Whether to compute surface metrics
        USE_TTA: Whether to use Test-Time Augmentation
        TTA_ANGLE_MAX: Maximum angle for TTA
        batch_size: Batch size
        patch_size: Patch size for inference
        tta_forward_fn: Function for Test-Time Augmentation
        sw_overlap: Overlap ratio for sliding window (0-1)
        
    Returns:
        dict: Dictionary with metrics
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_masd = 0.0
    running_nsd  = 0.0
    count_samples = 0
    
    original_patch_size = patch_size
    
    tta_transforms = get_tta_transforms(angle_max=TTA_ANGLE_MAX) if USE_TTA else None

    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                try:
                    inputs, labels = sample
                    inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
                    
                    if training_mode == "patch":
                        input_shape = inputs.shape[2:]  # [D, H, W]
                        if any(p >= s for p, s in zip(patch_size, input_shape)):
                            adjusted_patch_size = [min(p, s) for p, s in zip(patch_size, input_shape)]
                            patch_size = adjusted_patch_size
                    
                    if training_mode == "patch":
                        try:
                            logits = optimal_sliding_window_inference(
                                model=model, 
                                inputs=inputs, 
                                roi_size=patch_size, 
                                sw_batch_size=1, 
                                overlap=sw_overlap,
                                mode="gaussian",  
                                device=device
                            )
                        except Exception as e:
                            print(f"Error during sliding window inference: {e}")
                            print("Trying to process the input at once...")
                            logits = model(inputs)
                    else:
                        logits = model(inputs)

                    loss = loss_fn(logits, labels)
                    running_loss += loss.item()

                    if USE_TTA and tta_forward_fn is not None:
                        try:
                            avg_probs = tta_forward_fn(model, inputs, device, tta_transforms)
                            pred = np.argmax(avg_probs, axis=0)
                        except Exception as e:
                            print(f"Error during TTA: {e}. Using inference without TTA.")
                            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                    else:
                        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

                    label = labels.cpu().numpy()[0]
                    dsc = dice_coefficient(pred, label)
                    running_dice += dsc

                    if compute_surface_metrics:
                        try:
                            from ..utils.metrics import compute_masd, compute_nsd
                            masd = compute_masd(pred, label, spacing=(1,1,1), sampling_ratio=0.5)
                            nsd  = compute_nsd(pred, label, spacing=(1,1,1), sampling_ratio=0.5)
                            running_masd += masd
                            running_nsd  += nsd
                        except Exception as e:
                            print(f"Error computing surface metrics: {e}")
                            if count_samples == 0:  
                                print(f"Shapes: pred {pred.shape}, label {label.shape}")
                                
                    count_samples += 1
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
    
    patch_size = original_patch_size
    
    avg_loss = running_loss / count_samples if count_samples > 0 else 0.0
    avg_dice = running_dice / count_samples if count_samples > 0 else 0.0

    metrics = {'val_loss': avg_loss, 'val_dice': avg_dice}
    if compute_surface_metrics and count_samples > 0 and running_masd > 0:
        metrics['val_masd'] = running_masd / count_samples
        metrics['val_nsd']  = running_nsd  / count_samples

    return metrics


def setup_training(config, dataset_train, dataset_val, device="cuda"):
    """
    Prepares everything needed for training (model, optimizer, loss_fn, scheduler).
    
    Args:
        config: Configuration dictionary
        dataset_train: Training dataset
        dataset_val: Validation dataset
        device: Computation device
        
    Returns:
        tuple: (model, optimizer, loss_fn, scheduler, train_loader, val_loader)
    """
    train_loader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
    
    val_loader = DataLoader(
        dataset_val, 
        batch_size=config.get("val_batch_size", 2),  
        shuffle=False,
        collate_fn=custom_collate_fn  
    )
    
    model = create_model(
        model_name=config["model_name"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        drop_rate=config.get("drop_rate", 0.15)
    ).to(device)
    
    from ..loss import get_loss_function
    
    class_weights = None
    if config["loss_name"] == "weighted_ce_dice":
        class_weights = torch.tensor(
            [config["bg_weight"], config["fg_weight"]],
            device=device,
            dtype=torch.float
        )
    
    loss_fn = get_loss_function(
        loss_name=config["loss_name"],
        alpha=config["alpha"],
        class_weights=class_weights,
        focal_alpha=config.get("focal_alpha", 0.75),
        focal_gamma=config.get("focal_gamma", 2.0),
        alpha_mix=config.get("alpha_mix", 0.6),
        out_channels=config["out_channels"]
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["eta_min"]
    )
    
    return model, optimizer, loss_fn, scheduler, train_loader, val_loader


def train_with_ohem(model, dataset_train, optimizer, loss_fn, batch_size=1, device="cuda", ohem_ratio=0.15):
    """
    Trains the model using Online Hard Example Mining (OHEM).
    OHEM selects patches that are most difficult for the model (have the highest loss).
    
    Args:
        model: Model to train
        dataset_train: Dataset with training data
        optimizer: Optimizer
        loss_fn: Loss function
        batch_size: Batch size
        device: Computation device
        ohem_ratio: Ratio of hard examples to select (e.g., 0.15 = 15% of the hardest patches)
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    max_samples = 5000  
    sample_limit = min(len(dataset_train), max_samples)
    weighted_subset_size = min(int(sample_limit * ohem_ratio), batch_size * 100)  
    
    print(f"OHEM: Analyzing {sample_limit} samples from total {len(dataset_train)}")
    
    loss_per_sample = []
    indices = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for idx in range(sample_limit):
        try:
            inputs, labels = dataset_train[idx]
            inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                loss_per_sample.append(loss.item())
                indices.append(idx)
                
            del inputs, labels, logits, loss
            if idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during OHEM analysis of sample {idx}: {e}")
            continue
    
    if not loss_per_sample:
        print("Warning: No samples were successfully analyzed for OHEM. Skipping OHEM training.")
        return 0.0
    
    sorted_indices = [indices[i] for i in np.argsort(loss_per_sample)[::-1]]
    hard_indices = sorted_indices[:weighted_subset_size]
    
    print(f"OHEM: Selected {len(hard_indices)} hard samples for training")
    
    from torch.utils.data import Subset, DataLoader
    hard_subset = Subset(dataset_train, hard_indices)
    weighted_train_loader = DataLoader(hard_subset, batch_size=batch_size, shuffle=True)
    
    running_loss = 0.0
    batch_count = 0
    
    model.train()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for inputs, labels in weighted_train_loader:
        try:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            del inputs, labels, outputs, loss
            if batch_count % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during OHEM training on batch: {e}")
            continue
    
    return running_loss / max(1, batch_count)


def create_cv_folds(dataset, n_folds, extended_dataset=False):
    """
    Divides the dataset into k-fold cross-validation sets.
    
    Args:
        dataset: Dataset to divide
        n_folds: Number of folds
        extended_dataset: Whether this is an extended dataset with orig/aug files
        
    Returns:
        list: List of pairs (train_indices, val_indices)
    """
    all_files = dataset.adc_files
    
    subject_to_indices = {}
    for i, fname in enumerate(all_files):
        subj_id = get_subject_id_from_filename(fname)
        if subj_id not in subject_to_indices:
            subject_to_indices[subj_id] = []
        subject_to_indices[subj_id].append(i)
    
    all_subjects = list(subject_to_indices.keys())
    np.random.shuffle(all_subjects)
    
    num_subjects = len(all_subjects)
    fold_size = num_subjects // n_folds
    
    print(f"Total {num_subjects} unique subjects, creating {n_folds}-fold cross-validation.\n")
    
    folds = []
    for fold_idx in range(n_folds):
        fold_val_start = fold_idx * fold_size
        fold_val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else num_subjects
        val_subjects = all_subjects[fold_val_start:fold_val_end]
        train_subjects = list(set(all_subjects) - set(val_subjects))
        
        val_indices = []
        if extended_dataset:
            for subj_id in val_subjects:
                indices_for_subject = subject_to_indices[subj_id]
                for idx in indices_for_subject:
                    adc_fname = all_files[idx]
                    if "_aug" not in adc_fname.lower():
                        val_indices.append(idx)
        else:
            for subj_id in val_subjects:
                indices_for_subject = subject_to_indices[subj_id]
                val_indices.extend(indices_for_subject)
        
        train_indices = []
        for subj_id in train_subjects:
            indices_for_subject = subject_to_indices[subj_id]
            train_indices.extend(indices_for_subject)
        
        print(f"Fold {fold_idx+1}: Val subjects={len(val_subjects)}, Val samples={len(val_indices)}, Train subjects={len(train_subjects)}, Train samples={len(train_indices)}")
        folds.append((train_indices, val_indices))
    
    return folds


def log_metrics(metrics, epoch, fold_idx=None, lr=None, wandb_enabled=True):
    """
    Logs metrics to console and to wandb.
    
    Args:
        metrics: Dictionary with metrics
        epoch: Current epoch
        fold_idx: Index of current fold (None for fixed split)
        lr: Current learning rate
        wandb_enabled: Whether wandb is enabled
    """
    log_str = ""
    if fold_idx is not None:
        log_str += f"[FOLD {fold_idx+1}] "
    else:
        log_str += "[FIXED SPLIT] "
    
    log_str += f"Epoch {epoch} => "
    
    for k, v in metrics.items():
        log_str += f"{k}={v:.4f}, "
    
    if lr is not None:
        log_str += f"lr={lr:.6f}"
    
    print(log_str)
    
    if wandb_enabled:
        wandb_log = metrics.copy()
        
        if fold_idx is not None:
            wandb_log['fold'] = fold_idx + 1
        else:
            wandb_log['fold'] = 0  # 0 indicates fixed split
        
        wandb_log['epoch'] = epoch
        
        if lr is not None:
            wandb_log['lr'] = lr
        
        wandb.log(wandb_log) 