import os
import sys
import argparse
import torch
import wandb
import random
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Subset

# Automatic login to Weights & Biases using Kaggle secrets (if running in Kaggle)
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            print("Successfully logged into wandb using Kaggle secrets!")
        else:
            print("Warning: WANDB_API_KEY not found in Kaggle secrets.")
            print("Wandb logging will be disabled.")
            os.environ['WANDB_MODE'] = 'disabled'
    except Exception as e:
        print(f"Failed to log into wandb: {e}")
        print("Wandb logging will be disabled.")
        os.environ['WANDB_MODE'] = 'disabled'

from src.config import get_default_config, parse_args_to_config
from src.data.dataset import BONBID3DFullVolumeDataset, BONBID3DPatchDataset, get_subject_id_from_filename
from src.training.train import (
    setup_training, 
    train_one_epoch, 
    validate_one_epoch, 
    log_metrics,
    create_cv_folds,
    train_with_ohem
)
from src.inference.inference import (
    infer_full_volume,
    infer_full_volume_moe,
    save_segmentation_with_metrics
)

def set_seed(seed):
    """Sets a fixed seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimal_patch_size(dataset, requested_patch_size, min_dim_size=16):
    """
    Determines the optimal patch size based on the dimensions of the data in the dataset.
    
    Args:
        dataset: Dataset from which to determine dimensions
        requested_patch_size: Requested patch size [D, H, W]
        min_dim_size: Minimum size for one dimension
        
    Returns:
        tuple: Optimal patch size [D, H, W]
    """
    if not dataset or len(dataset) == 0:
        print("Warning: Empty dataset, cannot determine optimal patch size.")
        return requested_patch_size
    
    try:
        # Get the first sample from the dataset
        sample_data, _ = dataset[0]
        
        # Get shape - assume [C, D, H, W]
        input_shape = sample_data.shape[1:]  # [D, H, W]
        
        print(f"Input data dimensions: {input_shape}")
        print(f"Requested patch size: {requested_patch_size}")
        
        # Check if patch size is larger than input dimensions
        optimal_patch = list(requested_patch_size)
        need_adjustment = False
        
        for i in range(len(input_shape)):
            if optimal_patch[i] > input_shape[i]:
                optimal_patch[i] = max(min_dim_size, input_shape[i])
                need_adjustment = True
        
        if need_adjustment:
            print(f"Adjusted patch size: {optimal_patch}")
        
        # Check if patch size is divisible by 16 (for SwinUNETR)
        for i in range(len(optimal_patch)):
            if optimal_patch[i] % 16 != 0:
                # Round down to the nearest multiple of 16
                optimal_patch[i] = (optimal_patch[i] // 16) * 16
                if optimal_patch[i] < min_dim_size:
                    optimal_patch[i] = min_dim_size
        
        # Additional check after adjustments
        if optimal_patch != list(requested_patch_size):
            print(f"Final patch size after adjustment to multiples of 16: {optimal_patch}")
        
        return tuple(optimal_patch)
    
    except Exception as e:
        print(f"Error determining optimal patch size: {e}")
        return requested_patch_size


def create_fixed_split(dataset, split_ratio=0.8, extended_dataset=False, split_seed=None):
    """
    Creates a fixed split of data into training and validation sets (without k-fold CV).
    
    Args:
        dataset: Dataset to split
        split_ratio: Ratio of training data (e.g., 0.8 = 80% train, 20% validation)
        extended_dataset: Whether this is an extended dataset with orig/aug files
        split_seed: Specific seed for splitting (if None, uses global seed)
        
    Returns:
        list: List with a single fold [train_indices, val_indices]
    """
    # Get all files
    all_files = dataset.adc_files
    
    # Split by subjects
    subject_to_indices = {}
    for i, fname in enumerate(all_files):
        subj_id = get_subject_id_from_filename(fname)
        if subj_id not in subject_to_indices:
            subject_to_indices[subj_id] = []
        subject_to_indices[subj_id].append(i)
    
    # List of all subjects (shuffled using the specified seed)
    all_subjects = list(subject_to_indices.keys())
    
    # If a custom seed for splitting is provided, use it
    if split_seed is not None:
        rng = np.random.RandomState(split_seed)
        rng.shuffle(all_subjects)
    else:
        np.random.shuffle(all_subjects)
    
    # Calculate the split point
    split_idx = int(len(all_subjects) * split_ratio)
    
    # Select subjects for training and validation
    train_subjects = all_subjects[:split_idx]
    val_subjects = all_subjects[split_idx:]
    
    # Select validation and training indices
    val_indices = []
    if extended_dataset:
        # If we have an extended dataset, select only files without "_aug" marking
        for subj_id in val_subjects:
            indices_for_subject = subject_to_indices[subj_id]
            for idx in indices_for_subject:
                adc_fname = all_files[idx]
                if "_aug" not in adc_fname.lower():
                    val_indices.append(idx)
    else:
        # Standard dataset: use all files of the subject
        for subj_id in val_subjects:
            indices_for_subject = subject_to_indices[subj_id]
            val_indices.extend(indices_for_subject)
    
    # Select training indices - all files of subjects that are not in validation
    train_indices = []
    for subj_id in train_subjects:
        indices_for_subject = subject_to_indices[subj_id]
        train_indices.extend(indices_for_subject)
    
    print(f"Fixed data split: {len(train_subjects)} training subjects ({len(train_indices)} samples), "
          f"{len(val_subjects)} validation subjects ({len(val_indices)} samples)")
    
    return [(train_indices, val_indices)]


def run_cross_validation(config):
    """
    Runs k-fold cross-validation or fixed data split.
    
    Args:
        config: Configuration dictionary with parameters
    """
    # Check if folders exist
    for folder in [config["adc_folder"], config["z_folder"], config["label_folder"]]:
        if not os.path.exists(folder):
            print(f"Error: Folder {folder} does not exist!")
            sys.exit(1)
    
    # Perform preprocessing if requested
    if config.get("preprocessing", False):
        from src.data.preprocessing import prepare_preprocessed_data
        
        # Set paths for preprocessed data
        preprocessed_adc = config.get("preprocessed_adc_folder", os.path.join(config["output_dir"], "preprocessed/1ADC_ss"))
        preprocessed_z = config.get("preprocessed_z_folder", os.path.join(config["output_dir"], "preprocessed/2Z_ADC"))
        preprocessed_label = config.get("preprocessed_label_folder", os.path.join(config["output_dir"], "preprocessed/3LABEL"))
        
        # Run preprocessing
        print("\nPerforming data preprocessing...")
        prepare_preprocessed_data(
            adc_folder=config["adc_folder"],
            z_folder=config["z_folder"],
            label_folder=config["label_folder"],
            output_adc=preprocessed_adc,
            output_z=preprocessed_z,
            output_label=preprocessed_label,
            normalize=config.get("use_normalization", False),
            allow_normalize_spacing=config.get("allow_normalize_spacing", False)
        )
        
        # Update paths to preprocessed data
        config["adc_folder"] = preprocessed_adc
        config["z_folder"] = preprocessed_z
        config["label_folder"] = preprocessed_label
        
        print("Preprocessing complete. Using preprocessed data for training.\n")
    
    # Initialize wandb if enabled
    if config["use_wandb"]:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config
        )
    
    # Set seed
    set_seed(config["seed"])
    
    # Create dataset
    full_dataset = BONBID3DFullVolumeDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        augment=False,
        allowed_patient_ids=config.get("allowed_patient_ids", None),
        extended_dataset=config["extended_dataset"],
        max_aug_per_orig=config.get("max_aug_per_orig", 0),
        use_z_adc=config["in_channels"] > 1
    )
    
    # Determine optimal patch size
    if config["training_mode"] == "patch":
        sample_dataset = full_dataset
        if len(sample_dataset) > 0:
            original_patch_size = config["patch_size"]
            optimal_patch_size = get_optimal_patch_size(sample_dataset, original_patch_size)
            
            # Update configuration with optimal patch size
            if optimal_patch_size != original_patch_size:
                print(f"Original patch_size {original_patch_size} was adjusted to {optimal_patch_size} for super-resolution data.")
                config["patch_size"] = optimal_patch_size
                
                # If using wandb, update configuration
                if config["use_wandb"]:
                    wandb.config.update({"patch_size": optimal_patch_size}, allow_val_change=True)
    
    # Create folds or fixed split
    if config.get("fixed_split", False):
        print("Using fixed data split (80/20) instead of k-fold cross-validation...")
        
        # Determine path to fixed split file
        split_file = config.get("fixed_split_file", None)
        if split_file is None:
            # Create filename based on seed
            split_seed = config.get("fixed_split_seed", config["seed"])
            split_file = os.path.join(config["output_dir"], f"fixed_split_seed{split_seed}.pkl")
        
        # If file exists, load the split
        if os.path.exists(split_file):
            print(f"Loading existing fixed split from file {split_file}")
            try:
                with open(split_file, 'rb') as f:
                    split_data = pickle.load(f)
                    train_indices = split_data['train']
                    val_indices = split_data['val']
                folds = [(train_indices, val_indices)]
                print(f"Loaded: {len(train_indices)} training and {len(val_indices)} validation samples")
            except Exception as e:
                print(f"Error loading fixed split: {e}")
                print("Creating new fixed split...")
                # Create a new fixed split
                split_seed = config.get("fixed_split_seed", config["seed"])
                folds = create_fixed_split(
                    full_dataset, 
                    split_ratio=0.8, 
                    extended_dataset=config["extended_dataset"],
                    split_seed=split_seed
                )
                
                # Save for later use
                train_indices, val_indices = folds[0]
                split_data = {'train': train_indices, 'val': val_indices}
                with open(split_file, 'wb') as f:
                    pickle.dump(split_data, f)
                print(f"Fixed split saved to file {split_file}")
        else:
            # Create a new fixed split
            print(f"Creating new fixed data split (will be saved to {split_file})...")
            split_seed = config.get("fixed_split_seed", config["seed"])
            folds = create_fixed_split(
                full_dataset, 
                split_ratio=0.8, 
                extended_dataset=config["extended_dataset"],
                split_seed=split_seed
            )
            
            # Save for later use
            train_indices, val_indices = folds[0]
            split_data = {'train': train_indices, 'val': val_indices}
            with open(split_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Fixed split saved to file {split_file}")
    else:
        # Standard k-fold CV
        print(f"Creating {config['n_folds']}-fold cross-validation...")
        folds = create_cv_folds(
            full_dataset, 
            config["n_folds"], 
            extended_dataset=config["extended_dataset"]
        )
    
    # Determine device
    device = torch.device(config["device"])
    
    # Optimize GPU memory
    if torch.cuda.is_available():
        print("Optimizing GPU memory usage...")
        torch.cuda.empty_cache()

        # If enough GPUs are available, we can set specific optimizations
        if torch.cuda.device_count() > 0:
            torch.backends.cudnn.benchmark = True  # Can speed up training
            
        # Print available GPU memory
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_memory_gb = free_memory / (1024 ** 3)
            print(f"GPU {i}: Free memory {free_memory_gb:.2f} GB")
    
    # Cycles through all folds
    all_fold_metrics = []
    
    # For fixed split there will be only one "fold"
    num_folds = 1 if config.get("fixed_split", False) else config["n_folds"]
    
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"\n========== {'FOLD ' + str(fold_idx+1) + '/' + str(num_folds) if not config.get('fixed_split', False) else 'FIXED SPLIT'} ==========")
        
        # Create training and validation dataset
        if config["training_mode"] == "patch":
            # Patch-based training
            train_dataset_full = BONBID3DFullVolumeDataset(
                adc_folder=config["adc_folder"],
                z_folder=config["z_folder"],
                label_folder=config["label_folder"],
                augment=config["use_augmentation"],
                extended_dataset=config["extended_dataset"],
                max_aug_per_orig=config.get("max_aug_per_orig", 0),
                use_z_adc=config["in_channels"] > 1,
                augmentation_type=config.get("augmentation_type", "soft"),
                use_inpainted_lesions=config.get("use_inpainted_lesions", False),
                inpaint_adc_folder=config.get("inpaint_adc_folder"),
                inpaint_z_folder=config.get("inpaint_z_folder"),
                inpaint_label_folder=config.get("inpaint_label_folder"),
                inpaint_probability=config.get("inpaint_probability", 0.2)
            )
            
            # Use indices with IndexedDatasetWrapper
            train_dataset_full = Subset(train_dataset_full, train_indices)
            
            train_dataset = BONBID3DPatchDataset(
                full_volume_dataset=train_dataset_full,
                patch_size=config["patch_size"],
                patches_per_volume=config["patches_per_volume"],
                augment=config["use_augmentation"],
                intelligent_sampling=config.get("intelligent_sampling", True),
                foreground_ratio=config.get("foreground_ratio", 0.7),
                augmentation_type=config.get("augmentation_type", "soft")
            )
            
            val_dataset_full = BONBID3DFullVolumeDataset(
                adc_folder=config["adc_folder"],
                z_folder=config["z_folder"],
                label_folder=config["label_folder"],
                augment=False,
                extended_dataset=config["extended_dataset"],
                max_aug_per_orig=config.get("max_aug_per_orig", 0),
                use_z_adc=config["in_channels"] > 1
            )
            
            # Use indices with IndexedDatasetWrapper
            val_dataset = Subset(val_dataset_full, val_indices)
        else:
            # Full-volume training
            train_dataset_full = BONBID3DFullVolumeDataset(
                adc_folder=config["adc_folder"],
                z_folder=config["z_folder"],
                label_folder=config["label_folder"],
                augment=config["use_augmentation"],
                extended_dataset=config["extended_dataset"],
                max_aug_per_orig=config.get("max_aug_per_orig", 0),
                use_z_adc=config["in_channels"] > 1,
                augmentation_type=config.get("augmentation_type", "soft"),
                use_inpainted_lesions=config.get("use_inpainted_lesions", False),
                inpaint_adc_folder=config.get("inpaint_adc_folder"),
                inpaint_z_folder=config.get("inpaint_z_folder"),
                inpaint_label_folder=config.get("inpaint_label_folder"),
                inpaint_probability=config.get("inpaint_probability", 0.2)
            )
            
            # Use indices with IndexedDatasetWrapper
            train_dataset = Subset(train_dataset_full, train_indices)
            
            val_dataset_full = BONBID3DFullVolumeDataset(
                adc_folder=config["adc_folder"],
                z_folder=config["z_folder"],
                label_folder=config["label_folder"],
                augment=False,
                extended_dataset=config["extended_dataset"],
                max_aug_per_orig=config.get("max_aug_per_orig", 0),
                use_z_adc=config["in_channels"] > 1
            )
            
            # Use indices with IndexedDatasetWrapper
            val_dataset = Subset(val_dataset_full, val_indices)
        
        # Setup training
        model, optimizer, loss_fn, scheduler, train_loader, val_loader = setup_training(
            config, train_dataset, val_dataset, device=device
        )
        
        # Save best model
        best_val_dice = 0.0
        fold_metrics = {}
        
        # Training cycle
        for epoch in range(1, config["epochs"] + 1):
            # Train one epoch
            if config["use_ohem"] and epoch >= config["ohem_start_epoch"]:
                train_loss = train_with_ohem(
                    model=model,
                    dataset_train=train_dataset,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    batch_size=config["batch_size"],
                    device=device,
                    ohem_ratio=config.get("ohem_ratio", 0.15)
                )
            else:
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device
                )
            
            # Validation
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                training_mode=config["training_mode"],
                compute_surface_metrics=config["compute_surface_metrics"],
                USE_TTA=config["use_tta"],
                TTA_ANGLE_MAX=config["tta_angle_max"],
                batch_size=config["batch_size"],
                patch_size=config["patch_size"],
                sw_overlap=config.get("sw_overlap", 0.5)
            )
            
            # Update learning rate
            curr_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                **val_metrics
            }
            
            log_metrics(
                metrics=metrics,
                epoch=epoch,
                fold_idx=None if config.get("fixed_split", False) else fold_idx,
                lr=curr_lr,
                wandb_enabled=config["use_wandb"]
            )
            
            # Save best model
            if val_metrics['val_dice'] > best_val_dice:
                best_val_dice = val_metrics['val_dice']
                fold_metrics = val_metrics.copy()
                
                if config["save_best_model"]:
                    os.makedirs(config["model_dir"], exist_ok=True)
                    model_path = os.path.join(
                        config["model_dir"],
                        f"best_model_fold{fold_idx+1}.pth"
                    )
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved best model for fold {fold_idx+1}: {model_path}")
            
            # Inference on validation data (every N epochs)
            if config["inference_every_n_epochs"] > 0 and epoch % config["inference_every_n_epochs"] == 0:
                if val_indices:
                    # Create output directory for this epoch
                    output_dir = os.path.join(config["output_dir"], f"fold{fold_idx+1}", f"epoch{epoch}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    print(f"\nPerforming validation inference for all samples in epoch {epoch}:")
                    
                    # Check if saved best model exists
                    best_model_path = os.path.join(config["model_dir"], f"best_model_fold{fold_idx+1}.pth")
                    if os.path.exists(best_model_path):
                        print(f"  Loading best model from: {best_model_path}")
                        # Save current model state for later restoration
                        current_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        
                        # Load best model
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        print(f"  Using best model with DICE={best_val_dice:.4f} for inference")
                        using_best_model = True
                    else:
                        print(f"  Best model not found, using current model from epoch {epoch}")
                        using_best_model = False
                    
                    # Process all validation samples
                    for val_idx_pos, val_idx in enumerate(val_indices):
                        print(f"  Sample {val_idx_pos+1}/{len(val_indices)} (index {val_idx})")
                        
                        # Paths to data
                        adc_path = os.path.join(config["adc_folder"], full_dataset.adc_files[val_idx])
                        z_path = os.path.join(config["z_folder"], full_dataset.z_files[val_idx])
                        label_path = os.path.join(config["label_folder"], full_dataset.lab_files[val_idx])
                        
                        # Perform inference
                        result = infer_full_volume(
                            model=model,
                            input_paths=[adc_path, z_path],
                            label_path=label_path,
                            device=device,
                            use_tta=config["use_tta"],
                            tta_angle_max=config["tta_angle_max"],
                            training_mode=config["training_mode"],
                            patch_size=config["patch_size"],
                            batch_size=config["batch_size"],
                            use_z_adc=config["in_channels"] > 1
                        )
                        
                        # Print metrics
                        if result["metrics"]:
                            metrics = result["metrics"]
                            patient_id = result.get('patient_id', f'idx_{val_idx}')
                            print(f"    Patient {patient_id}: Dice={metrics['dice']:.4f}, MASD={metrics['masd']:.4f}, NSD={metrics['nsd']:.4f}")
                        
                        # Save segmentation and visualizations using new function for 3 columns
                        from src.inference import save_validation_results_pdf
                        
                        # Modify prefix to reflect that it is the best model
                        prefix = f"best_val_{val_idx_pos}" if using_best_model else f"val_{val_idx_pos}"
                        
                        # Save MHA file and standard PDF
                        save_segmentation_with_metrics(result, output_dir, prefix=prefix, save_pdf_comparison=False)
                        
                        # Save new PDF with three columns (ZADC, LABEL, PRED)
                        pdf_prefix = f"best_val3col_{val_idx_pos}" if using_best_model else f"val3col_{val_idx_pos}"
                        save_validation_results_pdf(result, output_dir, prefix=pdf_prefix)
                    
                    # Restore model to original state if best model was loaded
                    if using_best_model:
                        print("  Restoring original model state from current epoch")
                        model.load_state_dict(current_model_state)
                    
                    print(f"Validation inference complete. Results are saved in: {output_dir}")
        
        # Save metrics for fold
        all_fold_metrics.append(fold_metrics)
        
        # Print results for this fold
        if config.get("fixed_split", False):
            print(f"\nResults for FIXED SPLIT:")
        else:
            print(f"\nResults for FOLD {fold_idx+1}:")
            
        for k, v in fold_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Print average results across all folds
    if not config.get("fixed_split", False) and len(all_fold_metrics) > 1:
        print("\n========== OVERALL RESULTS ==========")
        avg_metrics = {}
        for metric in all_fold_metrics[0].keys():
            avg_value = sum(fold[metric] for fold in all_fold_metrics) / len(all_fold_metrics)
            avg_metrics[metric] = avg_value
            print(f"Average {metric}: {avg_value:.4f}")
        
        # Log final metrics to wandb
        if config["use_wandb"]:
            for k, v in avg_metrics.items():
                wandb.run.summary[f"avg_{k}"] = v
    else:
        # Fixed split or only one fold
        avg_metrics = all_fold_metrics[0]
        
        if config["use_wandb"]:
            for k, v in avg_metrics.items():
                wandb.run.summary[k] = v
    
    # Finish wandb
    if config["use_wandb"]:
        wandb.finish()
    
    return avg_metrics


def run_inference(config):
    """
    Runs inference on the specified data.
    
    Args:
        config: Configuration dictionary with parameters
    """
    # Check if folders exist
    for folder in [config["adc_folder"], config["z_folder"]]:
        if not os.path.exists(folder):
            print(f"Error: Folder {folder} does not exist!")
            sys.exit(1)
    
    # Perform preprocessing if requested
    if config.get("preprocessing", False):
        from src.data.preprocessing import prepare_preprocessed_data
        
        # Set paths for preprocessed data
        preprocessed_adc = config.get("preprocessed_adc_folder", os.path.join(config["output_dir"], "preprocessed/1ADC_ss"))
        preprocessed_z = config.get("preprocessed_z_folder", os.path.join(config["output_dir"], "preprocessed/2Z_ADC"))
        preprocessed_label = None
        if config.get("label_folder"):
            preprocessed_label = config.get("preprocessed_label_folder", os.path.join(config["output_dir"], "preprocessed/3LABEL"))
        
        # Run preprocessing
        print("\nPerforming data preprocessing...")
        prepare_preprocessed_data(
            adc_folder=config["adc_folder"],
            z_folder=config["z_folder"],
            label_folder=config.get("label_folder"),
            output_adc=preprocessed_adc,
            output_z=preprocessed_z,
            output_label=preprocessed_label if preprocessed_label else None,
            normalize=config.get("use_normalization", False),
            allow_normalize_spacing=config.get("allow_normalize_spacing", False)
        )
        
        # Update paths to preprocessed data
        config["adc_folder"] = preprocessed_adc
        config["z_folder"] = preprocessed_z
        if config.get("label_folder") and preprocessed_label:
            config["label_folder"] = preprocessed_label
        
        print("Preprocessing complete. Using preprocessed data for inference.\n")
    
    # Check if model path exists
    if not os.path.exists(config["model_path"]):
        print(f"Error: Model {config['model_path']} does not exist!")
        sys.exit(1)
    
    # Load file list
    adc_files = sorted([f for f in os.listdir(config["adc_folder"]) 
                        if f.endswith(".mha") or f.endswith(".nii") or f.endswith(".nii.gz")])
    z_files = sorted([f for f in os.listdir(config["z_folder"]) 
                      if f.endswith(".mha") or f.endswith(".nii") or f.endswith(".nii.gz")])
    
    if len(adc_files) != len(z_files):
        print("Error: Number of ADC and Z-ADC files does not match!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Determine device
    device = torch.device(config["device"])
    
    # Create and load model
    from src.models import create_model
    
    model = create_model(
        model_name=config["model_name"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        drop_rate=config.get("drop_rate", 0.15)
    ).to(device)
    
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()
    
    # Load expert model if using MoE inference
    expert_model = None
    if config["inference_mode"] == "moe" and config["expert_model_path"]:
        expert_model = create_model(
            model_name=config["expert_model_name"] or config["model_name"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            drop_rate=config.get("drop_rate", 0.15)
        ).to(device)
        expert_model.load_state_dict(torch.load(config["expert_model_path"], map_location=device))
        expert_model.eval()
    
    # Inference on all files
    for idx, (adc_file, z_file) in enumerate(zip(adc_files, z_files)):
        adc_path = os.path.join(config["adc_folder"], adc_file)
        z_path = os.path.join(config["z_folder"], z_file)
        
        # Check if corresponding ground truth exists
        label_path = None
        if config["label_folder"] and os.path.exists(config["label_folder"]):
            potential_label_files = [
                os.path.join(config["label_folder"], adc_file),
                os.path.join(config["label_folder"], adc_file.replace("ADC", "LABEL")),
                os.path.join(config["label_folder"], adc_file.replace("adc", "label"))
            ]
            for path in potential_label_files:
                if os.path.exists(path):
                    label_path = path
                    break
        
        print(f"\nInference for sample {idx+1}/{len(adc_files)}: {adc_file}")
        
        if config["inference_mode"] == "moe" and expert_model is not None:
            # MoE inference
            result = infer_full_volume_moe(
                main_model=model,
                expert_model=expert_model,
                input_paths=[adc_path, z_path],
                label_path=label_path,
                device=device,
                threshold=config["moe_threshold"],
                use_z_adc=config["in_channels"] > 1
            )
            
            # Print information about the used model
            print(f"Used model: {result['model_used']}, foreground voxels: {result['foreground_voxels']}")
            
        else:
            # Standard inference
            result = infer_full_volume(
                model=model,
                input_paths=[adc_path, z_path],
                label_path=label_path,
                device=device,
                use_tta=config["use_tta"],
                tta_angle_max=config["tta_angle_max"],
                training_mode=config["training_mode"],
                patch_size=config["patch_size"],
                batch_size=config["batch_size"],
                use_z_adc=config["in_channels"] > 1
            )
        
        # Print metrics if available
        if label_path and result["metrics"]:
            metrics = result["metrics"]
            print(f"Metrics: Dice={metrics['dice']:.4f}, MASD={metrics['masd']:.4f}, NSD={metrics['nsd']:.4f}")
        
        # Save results
        output_dir = config["output_dir"]
        save_segmentation_with_metrics(result, output_dir, save_pdf_comparison=config["save_pdf_comparison"])
    
    print(f"\nInference complete. Results are saved in: {config['output_dir']}")


def main():
    """Main program function"""
    # Get default configuration
    config = get_default_config()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="SWINUNetR for HIE lesion segmentation")
    
    # Run mode
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Run mode (train or inference)")
    
    # Dataset arguments
    parser.add_argument("--adc_folder", type=str, help="Path to folder with ADC images")
    parser.add_argument("--z_folder", type=str, help="Path to folder with Z-ADC images")
    parser.add_argument("--label_folder", type=str, help="Path to folder with ground truth masks")
    parser.add_argument("--preprocessed_adc_folder", type=str, help="Path to preprocessed ADC folder")
    parser.add_argument("--preprocessed_z_folder", type=str, help="Path to preprocessed Z_ADC folder")
    parser.add_argument("--preprocessed_label_folder", type=str, help="Path to preprocessed LABEL folder")
    parser.add_argument("--extended_dataset", action="store_true", 
                        help="Use extended dataset (with aug/orig files)")
    parser.add_argument("--max_aug_per_orig", type=int, default=0,
                        help="Maximum number of augmented files per original file")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--model_path", type=str, help="Path to saved model for inference")
    parser.add_argument("--expert_model_path", type=str, help="Path to expert model for MoE inference")
    parser.add_argument("--in_channels", type=int, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, help="Number of output classes")
    parser.add_argument("--drop_rate", type=float, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--eta_min", type=float, help="Minimum learning rate for scheduler")
    parser.add_argument("--training_mode", type=str, choices=["patch", "full_volume"], 
                        help="Training mode (patch or full_volume)")
    parser.add_argument("--no_augmentation", action="store_false", dest="use_augmentation",
                        help="Turn off data augmentation")
    parser.add_argument("--augmentation_type", type=str, choices=["soft", "heavy"], default="soft",
                        help="Type of augmentation for training ('soft' for light augmentations, 'heavy' for stronger augmentations)")
    parser.add_argument("--n_folds", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--use_ohem", action="store_true", help="Enable Online Hard Example Mining")
    parser.add_argument("--ohem_ratio", type=float, default=0.15, help="Ratio of hard examples for OHEM")
    parser.add_argument("--ohem_start_epoch", type=int, default=1, help="Epoch to start using OHEM")
    parser.add_argument("--patches_per_volume", type=int, help="Number of patches per volume in patch-based training")
    parser.add_argument("--patch_size", type=int, nargs=3, help="Patch size (3 values: height, width, depth)")
    parser.add_argument("--inference_every_n_epochs", type=int, default=0, help="Perform inference every N epochs (0 = disabled)")
    
    # Additional arguments for patch-based training
    parser.add_argument("--intelligent_sampling", action="store_true", help="Enable intelligent patch sampling focused on lesions")
    parser.add_argument("--foreground_ratio", type=float, default=0.7, help="Ratio of patches that should contain lesions (0-1)")
    parser.add_argument("--sw_overlap", type=float, default=0.5, help="Overlap ratio for sliding window inference (0-1)")
    
    # Inference arguments
    parser.add_argument("--inference_mode", type=str, choices=["standard", "moe"], default="standard", 
                        help="Inference mode (standard or moe)")
    parser.add_argument("--no_tta", action="store_false", dest="use_tta",
                        help="Turn off Test-Time Augmentation during inference")
    parser.add_argument("--moe_threshold", type=int, help="Threshold for switching to expert model")
    parser.add_argument("--save_pdf_comparison", action="store_true",
                        help="Create PDF with comparison of ground truth and prediction (only when label is available)")
    
    # General arguments
    parser.add_argument("--seed", type=int, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, help="Device for computation (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    # Data parameters and normalization
    parser.add_argument("--use_normalization", action="store_true", 
                        help="Use data normalization")
    parser.add_argument("--allow_normalize_spacing", action="store_true", 
                        help="Allow normalization of voxel spacing")
    parser.add_argument("--preprocessing", action="store_true",
                        help="Perform data preprocessing (bounding box, crop, padding) before training")
    
    # Fixed data split parameters
    parser.add_argument("--fixed_split", action="store_true",
                        help="Use fixed data split instead of k-fold CV (80/20)")
    parser.add_argument("--fixed_split_seed", type=int, default=None,
                        help="Seed for creating fixed data split (if different from main seed)")
    parser.add_argument("--fixed_split_file", type=str, default=None,
                        help="Path to file with saved fixed data split (.pkl)")

    # Loss and metrics parameters
    parser.add_argument("--loss_name", type=str, 
                        choices=["weighted_ce_dice", "log_cosh_dice", "focal_tversky", 
                                 "log_hausdorff", "focal", "focal_dice_combo", 
                                 "focal_ce_combo", "dice_focal", "weighted_ce"],
                        help="Loss function name")
    parser.add_argument("--focal_alpha", type=float, help="Alpha parameter for Focal loss")
    parser.add_argument("--focal_gamma", type=float, help="Gamma parameter for Focal loss")
    parser.add_argument("--alpha_mix", type=float, help="Mixing parameter for loss functions")
    parser.add_argument("--compute_surface_metrics", action="store_true", 
                        help="Compute surface metrics (MASD, NSD)")

    # TTA parameters
    parser.add_argument("--tta_angle_max", type=int, help="Maximum angle for TTA rotations")

    # Wandb arguments
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb",
                        help="Turn off logging to wandb")
    parser.add_argument("--wandb_project", type=str, help="Project name in wandb")
    parser.add_argument("--wandb_run_name", type=str, help="Run name in wandb")
    
    # Inpainted lesions
    parser.add_argument("--use_inpainted_lesions", action="store_true", 
                        help="Enable augmentation with inpainted lesions")
    parser.add_argument("--inpaint_adc_folder", type=str,
                        help="Folder with inpainted ADC files")
    parser.add_argument("--inpaint_z_folder", type=str,
                        help="Folder with inpainted Z-ADC files")
    parser.add_argument("--inpaint_label_folder", type=str,
                        help="Folder with inpainted LABEL files")
    parser.add_argument("--inpaint_probability", type=float, default=0.2,
                        help="Probability of using inpainted data (0.0-1.0)")

    # Update configuration from arguments
    args = parser.parse_args()
    config = parse_args_to_config(args, config)
    
    # Run requested mode
    if config["mode"] == "train":
        print("Starting training...")
        run_cross_validation(config)
    else:
        print("Starting inference...")
        run_inference(config)


if __name__ == "__main__":
    main() 