import os
import torch
from typing import Dict, Any, Optional

def get_default_config() -> Dict[str, Any]:
    """
    Returns the default configuration for training and inference.
    
    Returns:
        dict: Dictionary with default values
    """
    config = {
        # General parameters
        "mode": "train",                    # "train" or "inference"
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,                         # Seed for reproducibility
        "output_dir": "outputs",            # Output directory
        
        # Dataset parameters
        "adc_folder": None,                 # Path to folder with ADC images
        "z_folder": None,                   # Path to folder with Z-ADC images
        "label_folder": None,               # Path to folder with ground truth masks
        "allowed_patient_ids": None,        # List of allowed patient IDs
        "extended_dataset": False,          # Whether this is an extended dataset (with aug/orig files)
        
        # Model parameters
        "model_name": "swinunetr",          # Model name
        "in_channels": 2,                   # Number of input channels
        "out_channels": 2,                  # Number of output classes
        "model_path": None,                 # Path to saved model for inference
        "expert_model_path": None,          # Path to expert model for MoE
        "expert_model_name": None,          # Name of expert model
        "drop_rate": 0.15,                  # Dropout rate
        
        # Training parameters
        "batch_size": 1,                    # Batch size
        "epochs": 50,                       # Number of epochs
        "lr": 1e-4,                         # Learning rate
        "eta_min": 1e-6,                    # Minimum learning rate for scheduler
        "training_mode": "full_volume",     # "patch" or "full_volume"
        "patch_size": (64, 64, 64),         # Patch size for patch-based training
        "patches_per_volume": 4,            # Number of patches per volume
        "use_augmentation": True,           # Whether to use augmentation
        "use_ohem": False,                  # Whether to use Online Hard Example Mining
        "ohem_start_epoch": 10,             # Epoch to start using OHEM
        "n_folds": 5,                       # Number of folds for cross-validation
        "save_best_model": True,            # Whether to save the best model
        "model_dir": "models",              # Directory for saving models
        "compute_surface_metrics": True,    # Whether to compute surface metrics
        "inference_every_n_epochs": 0,      # Perform full-volume inference every N epochs
        
        # Loss parameters
        "loss_name": "log_cosh_dice",       # Loss function name
        "alpha": 0.5,                       # Weight for combined loss functions
        "alpha_mix": 0.6,                   # Mixing parameter for loss functions
        "focal_alpha": 0.75,                # Alpha parameter for Focal loss
        "focal_gamma": 2.0,                 # Gamma parameter for Focal loss
        "bg_weight": 1.0,                   # Background weight for weighted loss
        "fg_weight": 4.0,                   # Foreground weight for weighted loss
        
        # Inference parameters
        "inference_mode": "standard",       # "standard" or "moe"
        "use_tta": True,                    # Whether to use Test-Time Augmentation
        "tta_angle_max": 3,                 # Maximum angle for TTA rotations
        "moe_threshold": 80,                # Threshold for switching to expert model
        "save_pdf_comparison": True,        # Whether to create PDF comparisons of ground truth and prediction
        
        # Wandb parameters
        "use_wandb": False,                 # Whether to use Weights & Biases
        "wandb_project": "hie-segmentation", # Project name in wandb
        "wandb_run_name": None,             # Run name in wandb
    }
    
    # Automatic creation of run name if not provided
    if config["wandb_run_name"] is None:
        config["wandb_run_name"] = f"{config['model_name']}_{config['training_mode']}_{config['loss_name']}"
    
    return config

def parse_args_to_config(args, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Updates the configuration based on command-line arguments.
    
    Args:
        args: Arguments from argparse
        config: Existing configuration to update (optional)
        
    Returns:
        dict: Updated configuration
    """
    if config is None:
        config = get_default_config()
    
    # Update configuration from provided arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # Create output directories
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)
    
    if config.get("model_dir") and not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"], exist_ok=True)
    
    return config 