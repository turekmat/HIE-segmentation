import os
import argparse
from pathlib import Path

def get_default_config():
    """
    Vrací výchozí konfiguraci s parametry projektu
    """
    return {
        # Nastavení modelu
        "model_name": "SwinUNETR",  # SwinUNETR, UNet3Plus
        "in_channels": 2,
        "out_channels": 2,
        
        # Hyperparametry trénování
        "training_mode": "full_volume",  # patch, full_volume
        "use_moe": False,  # Mixture of Experts
        "n_folds": 5,
        "epochs": 15,
        "batch_size": 1,
        "lr": 0.0008,  # SwinBest 0.0008
        "eta_min": 4e-5,  # SwinBest 4e-6
        "step_size": 10,
        "gamma": 0.7,
        "drop_rate": 0.15,
        
        # Nastavení patch-based tréninku
        "patch_size": (64, 64, 64),
        "patch_per_volume": 1,
        
        # Nastavení ztráty
        "loss_name": "focal_dice_combo",  # weighted_ce_dice, log_cosh_dice, focal_tversky, log_hausdorff, focal, focal_dice_combo, focal_ce_combo, dice_focal
        "alpha": 0,
        "bg_weight": 1,
        "fg_weight": 40,
        "focal_alpha": 0.75,
        "focal_gamma": 2.0,
        "alpha_mix": 0.6,
        
        # Nastavení MoE modelu
        "expert_model_name": "SwinUNETR",
        "expert_loss_name": "focal_dice_combo",
        "expert_lr": 0.0004,
        "expert_eta_min": 1e-6,
        "threshold_expert": 60,
        
        # Nastavení augmentace a normalizace
        "use_data_augmentation": True,
        "use_normalization": True,
        "allow_normalize_spacing": True,
        "max_aug_per_orig": 0,
        "extended_dataset": True,
        
        # Test-time augmentace
        "USE_TTA": True,
        "TTA_ANGLE_MAX": 3,
        
        # Online hard example mining
        "use_ohem": False,
        "ohem_ratio": 0.15,
        
        # Metriky
        "compute_surface_metrics": True,
        
        # Cesty k adresářům
        "running_on": "local",  # local, kaggle, colab
        "data_paths": {
            "local": {
                "adc_folder": "./data/BONBID2023_Train/1ADC_ss",
                "z_folder": "./data/BONBID2023_Train/2Z_ADC",
                "label_folder": "./data/BONBID2023_Train/3LABEL",
                "preprocessed_adc_folder": "./data/preprocessed/1ADC_ss",
                "preprocessed_z_folder": "./data/preprocessed/2Z_ADC", 
                "preprocessed_label_folder": "./data/preprocessed/3LABEL",
                "out_dir_random": "./data/inference"
            },
            "kaggle": {
                "adc_folder": "/kaggle/input/bonbid-2023-train/BONBID2023_Train/1ADC_ss",
                "z_folder": "/kaggle/input/bonbid-2023-train/BONBID2023_Train/2Z_ADC",
                "label_folder": "/kaggle/input/bonbid-2023-train/BONBID2023_Train/3LABEL",
                "preprocessed_adc_folder": "/kaggle/working/preprocessed/1ADC_ss",
                "preprocessed_z_folder": "/kaggle/working/preprocessed/2Z_ADC",
                "preprocessed_label_folder": "/kaggle/working/preprocessed/3LABEL",
                "out_dir_random": "/kaggle/working/inference"
            },
            "colab": {
                "adc_folder": "/content/drive/MyDrive/data/BONBID2023_Train/1ADC_ss",
                "z_folder": "/content/drive/MyDrive/data/BONBID2023_Train/2Z_ADC",
                "label_folder": "/content/drive/MyDrive/data/BONBID2023_Train/3LABEL",
                "preprocessed_adc_folder": "/content/drive/MyDrive/data/preprocessed/BONBID2023_Train/1ADC_ss",
                "preprocessed_z_folder": "/content/drive/MyDrive/data/preprocessed/BONBID2023_Train/2Z_ADC",
                "preprocessed_label_folder": "/content/drive/MyDrive/data/preprocessed/BONBID2023_Train/3LABEL",
                "out_dir_random": "/content/drive/MyDrive/data/inference"
            }
        }
    }

def parse_args():
    """
    Parsuje argumenty příkazové řádky pro nastavení konfigurace
    """
    parser = argparse.ArgumentParser(description="HIE segmentace pomocí SwinUNETR")
    
    # Základní nastavení
    parser.add_argument("--running_on", type=str, choices=["local", "kaggle", "colab"], 
                        help="Prostředí, ve kterém kód běží (local, kaggle, colab)")
    parser.add_argument("--training_mode", type=str, choices=["full_volume", "patch"], 
                        help="Režim tréninku (full_volume nebo patch)")
    
    # Nastavení modelu
    parser.add_argument("--model_name", type=str, help="Jméno modelu (SwinUNETR, UNet3Plus)")
    parser.add_argument("--in_channels", type=int, help="Počet vstupních kanálů")
    parser.add_argument("--out_channels", type=int, help="Počet výstupních kanálů")
    
    # Hyperparametry trénování
    parser.add_argument("--use_moe", action="store_true", help="Použít Mixture of Experts")
    parser.add_argument("--n_folds", type=int, help="Počet foldů pro křížovou validaci")
    parser.add_argument("--epochs", type=int, help="Počet epoch")
    parser.add_argument("--batch_size", type=int, help="Velikost dávky")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--eta_min", type=float, help="Minimální learning rate pro scheduler")
    
    # Nastavení patch-based tréninku
    parser.add_argument("--patch_size", type=int, nargs=3, help="Velikost patche (x, y, z)")
    parser.add_argument("--patch_per_volume", type=int, help="Počet patchů na objem")
    
    # Nastavení ztráty
    parser.add_argument("--loss_name", type=str, help="Jméno ztrátové funkce")
    parser.add_argument("--alpha", type=float, help="Alpha parametr pro kombinované ztráty")
    parser.add_argument("--bg_weight", type=float, help="Váha pozadí")
    parser.add_argument("--fg_weight", type=float, help="Váha popředí")
    
    # Nastavení augmentace a normalizace
    parser.add_argument("--use_data_augmentation", action="store_true", help="Použít datovou augmentaci")
    parser.add_argument("--no_data_augmentation", action="store_false", dest="use_data_augmentation", 
                        help="Nepoužívat datovou augmentaci")
    parser.add_argument("--use_normalization", action="store_true", help="Použít normalizaci")
    parser.add_argument("--no_normalization", action="store_false", dest="use_normalization", 
                        help="Nepoužívat normalizaci")
    
    # Cesty k datům
    parser.add_argument("--adc_folder", type=str, help="Cesta k ADC složce")
    parser.add_argument("--z_folder", type=str, help="Cesta k Z_ADC složce")
    parser.add_argument("--label_folder", type=str, help="Cesta k LABEL složce")
    parser.add_argument("--preprocessed_adc_folder", type=str, help="Cesta k předzpracované ADC složce")
    parser.add_argument("--preprocessed_z_folder", type=str, help="Cesta k předzpracované Z_ADC složce")
    parser.add_argument("--preprocessed_label_folder", type=str, help="Cesta k předzpracované LABEL složce")
    parser.add_argument("--out_dir", type=str, help="Výstupní složka pro inference")
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """
    Aktualizuje konfiguraci podle argumentů z příkazové řádky
    """
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            if key == "patch_size" and value is not None:
                config[key] = tuple(value)
            elif key in ["adc_folder", "z_folder", "label_folder", 
                         "preprocessed_adc_folder", "preprocessed_z_folder", 
                         "preprocessed_label_folder", "out_dir"]:
                if "running_on" in config and config["running_on"] in config["data_paths"]:
                    if key == "out_dir":
                        config["data_paths"][config["running_on"]]["out_dir_random"] = value
                    else:
                        config["data_paths"][config["running_on"]][key] = value
            else:
                config[key] = value
    
    return config

def get_data_paths(config):
    """
    Vrací cesty k datům podle aktuálního nastavení running_on
    """
    running_on = config.get("running_on", "local")
    paths = config["data_paths"][running_on]
    
    # Pro extended_dataset upravujeme cesty k předzpracovaným datům
    if config.get("extended_dataset", False) and running_on in ["kaggle", "colab"]:
        if running_on == "kaggle":
            paths["preprocessed_adc_folder"] = "/kaggle/input/elastic-transform-bonbib/1ADC_ss"
            paths["preprocessed_z_folder"] = "/kaggle/input/elastic-transform-bonbib/2Z_ADC"
            paths["preprocessed_label_folder"] = "/kaggle/input/elastic-transform-bonbib/3LABEL"
        elif running_on == "colab":
            paths["preprocessed_adc_folder"] = "/content/drive/MyDrive/archive-3/1ADC_ss"
            paths["preprocessed_z_folder"] = "/content/drive/MyDrive/archive-3/2Z_ADC"
            paths["preprocessed_label_folder"] = "/content/drive/MyDrive/archive-3/3LABEL"
    
    return paths

def get_config():
    """
    Hlavní funkce pro získání konfigurace
    """
    config = get_default_config()
    args = parse_args()
    config = update_config_from_args(config, args)
    return config 