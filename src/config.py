import os
import torch
from typing import Dict, Any, Optional

def get_default_config() -> Dict[str, Any]:
    """
    Vrací výchozí konfiguraci pro trénování a inferenci.
    
    Returns:
        dict: Slovník s výchozími hodnotami
    """
    config = {
        # Obecné parametry
        "mode": "train",                    # "train" nebo "inference"
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,                         # Seed pro reprodukovatelnost
        "output_dir": "outputs",            # Výstupní adresář
        
        # Parametry datasetu
        "adc_folder": None,                 # Cesta ke složce s ADC snímky
        "z_folder": None,                   # Cesta ke složce s Z-ADC snímky
        "label_folder": None,               # Cesta ke složce s ground truth maskami
        "allowed_patient_ids": None,        # Seznam povolených ID pacientů
        "extended_dataset": False,          # Zda se jedná o rozšířený dataset (s aug/orig soubory)
        
        # Parametry modelu
        "model_name": "swinunetr",          # Jméno modelu
        "in_channels": 2,                   # Počet vstupních kanálů
        "out_channels": 2,                  # Počet výstupních tříd
        "model_path": None,                 # Cesta k uloženému modelu pro inference
        "expert_model_path": None,          # Cesta k expertnímu modelu pro MoE
        "expert_model_name": None,          # Jméno expertního modelu
        "drop_rate": 0.15,                  # Dropout rate
        
        # Parametry trénování
        "batch_size": 1,                    # Velikost dávky
        "epochs": 50,                       # Počet epoch
        "lr": 1e-4,                         # Learning rate
        "eta_min": 1e-6,                    # Minimální learning rate pro scheduler
        "training_mode": "full_volume",     # "patch" nebo "full_volume"
        "patch_size": (64, 64, 64),         # Velikost patche pro patch-based trénink
        "patches_per_volume": 4,            # Počet patchů na jeden volume
        "use_augmentation": True,           # Zda používat augmentaci
        "use_ohem": False,                  # Zda používat Online Hard Example Mining
        "ohem_start_epoch": 10,             # Od které epochy začít používat OHEM
        "n_folds": 5,                       # Počet foldů pro cross-validaci
        "save_best_model": True,            # Zda ukládat nejlepší model
        "model_dir": "models",              # Adresář pro ukládání modelů
        "compute_surface_metrics": True,    # Zda počítat povrchové metriky
        "inference_every_n_epochs": 0,      # Provádět full-volume inferenci každých N epoch
        
        # Parametry ztráty
        "loss_name": "log_cosh_dice",       # Jméno ztrátové funkce
        "alpha": 0.5,                       # Váha pro kombinované ztrátové funkce
        "alpha_mix": 0.6,                   # Míchací parametr pro ztrátové funkce
        "focal_alpha": 0.75,                # Alpha parametr pro Focal loss
        "focal_gamma": 2.0,                 # Gamma parametr pro Focal loss
        "bg_weight": 1.0,                   # Váha pozadí pro weighted loss
        "fg_weight": 4.0,                   # Váha popředí pro weighted loss
        
        # Parametry inference
        "inference_mode": "standard",       # "standard" nebo "moe"
        "use_tta": True,                    # Zda používat Test-Time Augmentation
        "tta_angle_max": 3,                 # Maximální úhel pro TTA rotace
        "moe_threshold": 80,                # Threshold pro přepnutí na expertní model
        "save_pdf_comparison": True,       # Zda vytvářet PDF s porovnáním ground truth a predikce
        
        # Wandb parametry
        "use_wandb": False,                 # Zda používat Weights & Biases
        "wandb_project": "hie-segmentation", # Jméno projektu ve wandb
        "wandb_run_name": None,             # Jméno běhu ve wandb
        
        # Parametry kaskádového přístupu
        "use_cascaded_approach": False,
        "cascaded_mode": "combined",  # "roi_only" nebo "combined"
        "small_lesion_model": "small_unet",
        "small_lesion_patch_size": [16, 16, 16],
        "small_lesion_threshold": 0.5,
        "small_lesion_max_voxels": 50,
        "small_lesion_model_path": None,
        "small_lesion_epochs": 50,
        "small_lesion_batch_size": 16,
        "small_lesion_lr": 0.001,
        "small_lesion_patches_per_volume": 200,
        "small_lesion_foreground_ratio": 0.8,
        "small_lesion_large_lesion_sampling_ratio": 0.25,
        "retrain_small_lesion_model": False,
        "preload_small_lesion_volumes": True,  # Zda předem načíst objemy do paměti pro trénink malého modelu
        
        # Parametry pro pokročilou kombinaci predikcí
        "advanced_combine": False,
        "combine_alpha": 0.6,
        "combine_boost_factor": 1.5,
        "combine_high_conf_threshold": 0.8,
        "combine_adaptive": True,
        "combine_size_based": True,
        
        # Parametry pro vylepšený kaskádový přístup s AttentionResUNet
        "use_enhanced_cascade": False,
        "use_cbam": True,
        "use_feature_fusion": True,
        "enhanced_model_features": [32, 64, 128, 256],
    }
    
    # Automatické vytvoření jména běhu, pokud není uvedeno
    if config["wandb_run_name"] is None:
        config["wandb_run_name"] = f"{config['model_name']}_{config['training_mode']}_{config['loss_name']}"
    
    return config

def parse_args_to_config(args, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Aktualizuje konfiguraci na základě argumentů příkazové řádky.
    
    Args:
        args: Argumenty z argparse
        config: Existující konfigurace k aktualizaci (volitelné)
        
    Returns:
        dict: Aktualizovaná konfigurace
    """
    if config is None:
        config = get_default_config()
    
    # Aktualizace konfigurace ze zadaných argumentů
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # Speciální zpracování pro vzájemně vylučující se argumenty
    if hasattr(args, 'no_preload_small_lesion_volumes') and args.no_preload_small_lesion_volumes:
        config["preload_small_lesion_volumes"] = False
    elif hasattr(args, 'preload_small_lesion_volumes') and args.preload_small_lesion_volumes:
        config["preload_small_lesion_volumes"] = True
    
    # Vytvoření výstupních adresářů
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)
    
    if config.get("model_dir") and not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"], exist_ok=True)
    
    return config 