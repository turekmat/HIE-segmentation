import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk
import pickle
import random
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Automatické přihlášení k Weights & Biases pomocí Kaggle secrets (pokud běží v Kaggle)
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            print("Úspěšně přihlášeno do wandb pomocí Kaggle secrets!")
        else:
            print("Varování: WANDB_API_KEY nenalezen v Kaggle secrets.")
            print("Wandb logování bude vypnuto.")
            os.environ['WANDB_MODE'] = 'disabled'
    except Exception as e:
        print(f"Nepodařilo se přihlásit do wandb: {e}")
        print("Wandb logování bude vypnuto.")
        os.environ['WANDB_MODE'] = 'disabled'

from src.config import get_default_config, parse_args_to_config
from src.data.dataset import BONBID3DFullVolumeDataset, BONBID3DPatchDataset, get_subject_id_from_filename
from src.training.train import (
    setup_training, 
    train_one_epoch, 
    validate_one_epoch, 
    log_metrics,
    create_cv_folds,
    train_with_ohem,
)
from src.inference.inference import (
    infer_full_volume,
    infer_full_volume_moe,
    save_segmentation_with_metrics,
    infer_full_volume_cascaded,
    get_tta_transforms,
    save_validation_results_pdf
)
from src.models import create_model
from src.utils import setup_wandb
from src.models import create_small_lesion_model
from src.utils.metrics import compute_all_metrics

def set_seed(seed):
    """Nastavení fixního seedu pro reprodukovatelnost"""
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
    Určí optimální velikost patche na základě rozměrů dat v datasetu.
    
    Args:
        dataset: Dataset, ze kterého budeme zjišťovat rozměry
        requested_patch_size: Požadovaná velikost patche [D, H, W]
        min_dim_size: Minimální velikost pro jednu dimenzi
        
    Returns:
        tuple: Optimální velikost patche [D, H, W]
    """
    if not dataset or len(dataset) == 0:
        print("Varování: Prázdný dataset, nemohu určit optimální velikost patche.")
        return requested_patch_size
    
    try:
        # Získat první vzorek z datasetu
        sample_data, _ = dataset[0]
        
        # Zjistit shape - předpokládáme [C, D, H, W]
        input_shape = sample_data.shape[1:]  # [D, H, W]
        
        print(f"Rozměry vstupních dat: {input_shape}")
        print(f"Požadovaná velikost patche: {requested_patch_size}")
        
        # Kontrola, jestli patch size není větší než vstupní dimensions
        optimal_patch = list(requested_patch_size)
        need_adjustment = False
        
        for i in range(len(input_shape)):
            if optimal_patch[i] > input_shape[i]:
                optimal_patch[i] = max(min_dim_size, input_shape[i])
                need_adjustment = True
        
        if need_adjustment:
            print(f"Upravená velikost patche: {optimal_patch}")
        
        # Zkontrolovat, zda je velikost patche dělitelná 16 (pro SwinUNETR)
        for i in range(len(optimal_patch)):
            if optimal_patch[i] % 16 != 0:
                # Zaokrouhlit dolů na nejbližší násobek 16
                optimal_patch[i] = (optimal_patch[i] // 16) * 16
                if optimal_patch[i] < min_dim_size:
                    optimal_patch[i] = min_dim_size
        
        # Další kontrola po úpravách
        if optimal_patch != list(requested_patch_size):
            print(f"Finální velikost patche po úpravě na násobky 16: {optimal_patch}")
        
        return tuple(optimal_patch)
    
    except Exception as e:
        print(f"Chyba při určování optimální velikosti patche: {e}")
        return requested_patch_size


def create_fixed_split(dataset, split_ratio=0.8, extended_dataset=False, split_seed=None):
    """
    Vytvoří pevné rozdělení dat na trénovací a validační sadu (bez k-fold CV).
    
    Args:
        dataset: Dataset k rozdělení
        split_ratio: Poměr trénovacích dat (např. 0.8 = 80% train, 20% validation)
        extended_dataset: Zda se jedná o rozšířený dataset s orig/aug soubory
        split_seed: Specifický seed pro rozdělení (pokud None, použije se globální seed)
        
    Returns:
        list: Seznam s jedním foldem [train_indices, val_indices]
    """
    # Získání všech souborů
    all_files = dataset.adc_files
    
    # Rozdělení podle subjektů
    subject_to_indices = {}
    for i, fname in enumerate(all_files):
        subj_id = get_subject_id_from_filename(fname)
        if subj_id not in subject_to_indices:
            subject_to_indices[subj_id] = []
        subject_to_indices[subj_id].append(i)
    
    # Seznam všech subjektů (promíchaný s použitím zadaného seedu)
    all_subjects = list(subject_to_indices.keys())
    
    # Pokud je zadán vlastní seed pro rozdělení, použijeme ho
    if split_seed is not None:
        rng = np.random.RandomState(split_seed)
        rng.shuffle(all_subjects)
    else:
        np.random.shuffle(all_subjects)
    
    # Výpočet dělícího bodu pro split
    split_idx = int(len(all_subjects) * split_ratio)
    
    # Výběr subjektů pro training a validation
    train_subjects = all_subjects[:split_idx]
    val_subjects = all_subjects[split_idx:]
    
    # Výběr validačních a trénovacích indexů
    val_indices = []
    if extended_dataset:
        # Pokud máme rozšířený dataset, vybíráme pouze soubory bez "_aug" označení
        for subj_id in val_subjects:
            indices_for_subject = subject_to_indices[subj_id]
            for idx in indices_for_subject:
                adc_fname = all_files[idx]
                if "_aug" not in adc_fname.lower():
                    val_indices.append(idx)
    else:
        # Klasický dataset: použijeme všechny soubory daného subjektu
        for subj_id in val_subjects:
            indices_for_subject = subject_to_indices[subj_id]
            val_indices.extend(indices_for_subject)
    
    # Výběr trénovacích indexů - všechny soubory subjektů, které nejsou ve validaci
    train_indices = []
    for subj_id in train_subjects:
        indices_for_subject = subject_to_indices[subj_id]
        train_indices.extend(indices_for_subject)
    
    print(f"Pevné rozdělení dat: {len(train_subjects)} trénovacích subjektů ({len(train_indices)} vzorků), "
          f"{len(val_subjects)} validačních subjektů ({len(val_indices)} vzorků)")
    
    return [(train_indices, val_indices)]


def run_cross_validation(config):
    """
    Spustí k-fold cross-validaci nebo pevné rozdělení dat.
    
    Args:
        config: Konfigurační slovník s parametry
    """
    # Kontrola, zda složky existují
    for folder in [config["adc_folder"], config["z_folder"], config["label_folder"]]:
        if not os.path.exists(folder):
            print(f"Chyba: Složka {folder} neexistuje!")
            sys.exit(1)
    
    # Provedení preprocessingu, pokud je požadováno
    if config.get("preprocessing", False):
        from src.data.preprocessing import prepare_preprocessed_data
        
        # Nastavení cest pro předzpracovaná data
        preprocessed_adc = config.get("preprocessed_adc_folder", os.path.join(config["output_dir"], "preprocessed/1ADC_ss"))
        preprocessed_z = config.get("preprocessed_z_folder", os.path.join(config["output_dir"], "preprocessed/2Z_ADC"))
        preprocessed_label = config.get("preprocessed_label_folder", os.path.join(config["output_dir"], "preprocessed/3LABEL"))
        
        # Spuštění preprocessingu
        print("\nProvádím preprocessing dat...")
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
        
        # Aktualizace cest na předzpracovaná data
        config["adc_folder"] = preprocessed_adc
        config["z_folder"] = preprocessed_z
        config["label_folder"] = preprocessed_label
        
        print("Preprocessing dokončen. Používám předzpracovaná data pro trénink.\n")
    
    # Inicializace wandb, pokud je povoleno
    if config["use_wandb"]:
        setup_wandb(
            project_name=config["wandb_project"],
            run_name=config["wandb_run_name"],
            config=config
        )
    
    # Nastavení seedu
    set_seed(config["seed"])
    
    # Vytvoření datasetu
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
    
    # Zjištění optimální velikosti patche
    if config["training_mode"] == "patch":
        sample_dataset = full_dataset
        if len(sample_dataset) > 0:
            original_patch_size = config["patch_size"]
            optimal_patch_size = get_optimal_patch_size(sample_dataset, original_patch_size)
            
            # Aktualizace konfigurace s optimální velikostí patche
            if optimal_patch_size != original_patch_size:
                print(f"Původní patch_size {original_patch_size} byl upraven na {optimal_patch_size} pro super-resolution data.")
                config["patch_size"] = optimal_patch_size
                
                # Pokud používáme wandb, aktualizujeme konfiguraci
                if config["use_wandb"]:
                    wandb.config.update({"patch_size": optimal_patch_size}, allow_val_change=True)
    
    # Vytvoření foldů nebo pevného rozdělení
    if config.get("fixed_split", False):
        print("Používám pevné rozdělení dat (80/20) místo k-fold cross-validace...")
        
        # Určení cesty k souboru s pevným rozdělením
        split_file = config.get("fixed_split_file", None)
        if split_file is None:
            # Vytvoření jména souboru na základě seedu
            split_seed = config.get("fixed_split_seed", config["seed"])
            split_file = os.path.join(config["output_dir"], f"fixed_split_seed{split_seed}.pkl")
        
        # Pokud soubor existuje, načteme rozdělení
        if os.path.exists(split_file):
            print(f"Načítám existující pevné rozdělení ze souboru {split_file}")
            try:
                with open(split_file, 'rb') as f:
                    split_data = pickle.load(f)
                    train_indices = split_data['train']
                    val_indices = split_data['val']
                folds = [(train_indices, val_indices)]
                print(f"Načteno: {len(train_indices)} trénovacích a {len(val_indices)} validačních vzorků")
            except Exception as e:
                print(f"Chyba při načítání pevného rozdělení: {e}")
                print("Vytvářím nové pevné rozdělení...")
                # Vytvoření nového pevného rozdělení
                split_seed = config.get("fixed_split_seed", config["seed"])
                folds = create_fixed_split(
                    full_dataset, 
                    split_ratio=0.8, 
                    extended_dataset=config["extended_dataset"],
                    split_seed=split_seed
                )
                
                # Uložení pro pozdější použití
                train_indices, val_indices = folds[0]
                split_data = {'train': train_indices, 'val': val_indices}
                with open(split_file, 'wb') as f:
                    pickle.dump(split_data, f)
                print(f"Pevné rozdělení uloženo do souboru {split_file}")
        else:
            # Vytvoření nového pevného rozdělení
            print(f"Vytvářím nové pevné rozdělení dat (bude uloženo do {split_file})...")
            split_seed = config.get("fixed_split_seed", config["seed"])
            folds = create_fixed_split(
                full_dataset, 
                split_ratio=0.8, 
                extended_dataset=config["extended_dataset"],
                split_seed=split_seed
            )
            
            # Uložení pro pozdější použití
            train_indices, val_indices = folds[0]
            split_data = {'train': train_indices, 'val': val_indices}
            with open(split_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Pevné rozdělení uloženo do souboru {split_file}")
    else:
        # Standardní k-fold CV
        print(f"Vytváření {config['n_folds']}-fold cross-validace...")
        folds = create_cv_folds(
            full_dataset, 
            config["n_folds"], 
            extended_dataset=config["extended_dataset"]
        )
    
    # Určení device
    device = torch.device(config["device"])
    
    # Optimalizace GPU paměti
    if torch.cuda.is_available():
        print("Optimalizace využití GPU paměti...")
        torch.cuda.empty_cache()

        # Pokud je k dispozici dostatek GPU, můžeme nastavit specifické optimalizace
        if torch.cuda.device_count() > 0:
            torch.backends.cudnn.benchmark = True  # Může zrychlit trénink
            
        # Výpis dostupné paměti GPU
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_memory_gb = free_memory / (1024 ** 3)
            print(f"GPU {i}: Volná paměť {free_memory_gb:.2f} GB")
    
    # Cykly přes všechny foldy
    all_fold_metrics = []
    
    # Pro fixní rozdělení bude jen jeden "fold"
    num_folds = 1 if config.get("fixed_split", False) else config["n_folds"]
    
    # Uchováváme cesty ke všem natrénovaným modelům pro malé léze
    small_lesion_model_paths = {}
    
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"\n========== {'FOLD ' + str(fold_idx+1) + '/' + str(num_folds) if not config.get('fixed_split', False) else 'PEVNÉ ROZDĚLENÍ'} ==========")
        
        # Vytvoření trénovacího a validačního datasetu
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
                augmentation_type=config.get("augmentation_type", "soft")
            )
            
            # Použití indexů pomocí IndexedDatasetWrapper
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
            
            # Použití indexů pomocí IndexedDatasetWrapper
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
                augmentation_type=config.get("augmentation_type", "soft")
            )
            
            # Použití indexů pomocí IndexedDatasetWrapper
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
            
            # Použití indexů pomocí IndexedDatasetWrapper
            val_dataset = Subset(val_dataset_full, val_indices)
        
        # Trénink modelu pro detekci malých lézí pro tento fold, pokud je povolen kaskádový přístup
        if config.get("use_cascaded_approach", False):
            print(f"\nKaskádový přístup je povolen pro fold {fold_idx+1}")
            
            # Import funkce pro trénink modelu malých lézí
            from src.training import train_small_lesion_model, train_small_lesion_model_with_indices
            
            # Nastavení cesty pro ukládání modelu malých lézí pro tento fold
            small_lesion_model_dir = os.path.join(config.get("model_dir", config["output_dir"]), "small_lesion")
            os.makedirs(small_lesion_model_dir, exist_ok=True)
            
            # Určení cesty k souboru modelu pro tento fold
            small_lesion_model_path = os.path.join(small_lesion_model_dir, f"best_small_lesion_model_fold{fold_idx+1}.pth")
            
            # Kontrola, zda je soubor s modelem pro tento fold už trénovaný
            if os.path.exists(small_lesion_model_path) and not config.get("retrain_small_lesion_model", False):
                print(f"Načítám existující model pro malé léze pro fold {fold_idx+1} z: {small_lesion_model_path}")
            else:
                print(f"Trénuji model pro malé léze pro fold {fold_idx+1}...")
                
                # Trénink modelu malých lézí pouze na trénovacích datech tohoto foldu
                small_model_path = train_small_lesion_model_with_indices(
                    config, 
                    train_indices, 
                    fold_idx,
                    device=device
                )
                
                # Aktualizace cesty k modelu
                small_lesion_model_path = small_model_path
                
                print(f"Model pro malé léze (fold {fold_idx+1}) natrénován a uložen v: {small_lesion_model_path}")
            
            # Uložení cesty k modelu pro tento fold
            small_lesion_model_paths[fold_idx] = small_lesion_model_path
            
            # Nastavení cesty k modelu malých lézí v konfiguraci pro tento fold
            config["small_lesion_model_path"] = small_lesion_model_path
            
            # Aktualizace konfigurace ve wandb, pokud je povoleno
            if config["use_wandb"]:
                wandb.config.update({"small_lesion_model_path": small_lesion_model_path}, allow_val_change=True)
        
        # Nastavení trénování
        model, optimizer, loss_fn, scheduler, train_loader, val_loader = setup_training(
            config, train_dataset, val_dataset, device=device
        )
        
        # Ukládání nejlepšího modelu
        best_val_dice = 0.0
        fold_metrics = {}
        
        # Tréninkový cyklus
        for epoch in range(1, config["epochs"] + 1):
            # Trénování jedné epochy
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
            
            # Validace
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
            
            # Aktualizace learning rate
            curr_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Logování metrik
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
            
            # Ukládání nejlepšího modelu
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
                    print(f"Uložen nejlepší model pro fold {fold_idx+1}: {model_path}")
            
            # Inference na validačních datech (každých N epoch)
            if config["inference_every_n_epochs"] > 0 and epoch % config["inference_every_n_epochs"] == 0:
                if val_indices:
                    # Vytvoření výstupního adresáře pro tuto epochu
                    output_dir = os.path.join(config["output_dir"], f"fold{fold_idx+1}", f"epoch{epoch}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    print(f"\nProvádím validační inferenci pro všechny vzorky v epochě {epoch}:")
                    
                    # Kontrola, zda existuje uložený nejlepší model
                    best_model_path = os.path.join(config["model_dir"], f"best_model_fold{fold_idx+1}.pth")
                    if os.path.exists(best_model_path):
                        print(f"  Načítám nejlepší model z: {best_model_path}")
                        # Uložení aktuálního stavu modelu pro pozdější obnovení
                        current_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        
                        # Načtení nejlepšího modelu
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        print(f"  Používám nejlepší model s DICE={best_val_dice:.4f} pro inferenci")
                        using_best_model = True
                    else:
                        print(f"  Nejlepší model nenalezen, používám aktuální model z epochy {epoch}")
                        using_best_model = False
                    
                    # Kontrola, zda se má použít kaskádový přístup
                    if config.get("use_cascaded_approach", False):
                        # Ujistíme se, že máme cestu k modelu pro malé léze pro tento fold
                        if fold_idx in small_lesion_model_paths:
                            small_lesion_model_path = small_lesion_model_paths[fold_idx]
                            print(f"  Používám model pro malé léze pro fold {fold_idx+1} z: {small_lesion_model_path}")
                            config["small_lesion_model_path"] = small_lesion_model_path
                            
                            # Načtení modelu pro malé léze
                            small_lesion_model = create_small_lesion_model(
                                model_name=config["small_lesion_model"],
                                in_channels=config["in_channels"],
                                out_channels=config["out_channels"]
                            )
                            small_lesion_model.to(device)
                            small_lesion_model.load_state_dict(torch.load(small_lesion_model_path, map_location=device))
                            small_lesion_model.eval()
                            print(f"  Model pro malé léze úspěšně načten.")
                        else:
                            print(f"  Model pro malé léze pro fold {fold_idx+1} nenalezen! Používám výchozí model.")
                            # Vytvoření nového modelu
                            small_lesion_model = create_small_lesion_model(
                                model_name=config["small_lesion_model"],
                                in_channels=config["in_channels"],
                                out_channels=config["out_channels"]
                            )
                            small_lesion_model.to(device)
                            small_lesion_model.eval()
                            print(f"  VAROVÁNÍ: Používám nenatrénovaný model pro malé léze!")
                    else:
                        small_lesion_model = None
                    
                    # Načteme data validačního datasetu do paměti
                    val_data = []
                    
                    # Zpracujeme všechny validační vzorky
                    for val_idx_pos, val_idx in enumerate(val_indices):
                        print(f"  Vzorek {val_idx_pos+1}/{len(val_indices)} (index {val_idx})")
                        
                        # Cesty k datům
                        adc_path = os.path.join(config["adc_folder"], full_dataset.adc_files[val_idx])
                        z_path = os.path.join(config["z_folder"], full_dataset.z_files[val_idx])
                        label_path = os.path.join(config["label_folder"], full_dataset.lab_files[val_idx])
                        
                        # Provedení inference
                        input_paths = [adc_path, z_path]
                        # Načtení vstupních dat
                        volumes = []
                        
                        # Vždy načíst ADC mapu (první v seznamu)
                        adc_path_curr = input_paths[0]
                        sitk_img = sitk.ReadImage(adc_path_curr)
                        np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                        volumes.append(np_vol)
                        
                        # Načíst Z-ADC mapu, pouze pokud se používá
                        if config["in_channels"] > 1 and len(input_paths) > 1:
                            zadc_path = input_paths[1]
                            try:
                                sitk_zadc = sitk.ReadImage(zadc_path)
                                zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
                                volumes.append(zadc_np)
                            except Exception as e:
                                print(f"Varování: Nelze načíst Z-ADC soubor {zadc_path}: {e}")
                                print("Inference bude provedena pouze s ADC mapou.")
                        
                        # Vytvoření vstupního tensoru
                        input_vol = np.stack(volumes, axis=0)  # tvar: (C, D, H, W)
                        
                        # Získání transformací pro TTA, pokud je povoleno
                        tta_transforms = None
                        if config.get("use_tta", True):
                            tta_transforms = get_tta_transforms(angle_max=config.get("tta_angle_max", 3))
                        
                        # Ujistíme se, že small_lesion_model je definován
                        if not config.get("use_cascaded_approach", False) or 'small_lesion_model' not in locals():
                            small_lesion_model = None
                        
                        # Inference s aktuálním ensemblem modelů
                        result_item = infer_full_volume_cascaded(
                            input_vol=input_vol,
                            main_model=model,
                            small_lesion_model=small_lesion_model,
                            device=device,
                            use_tta=config.get("use_tta", True),
                            tta_transforms=tta_transforms,
                            cascaded_mode=config.get("cascaded_mode", "combined"),
                            small_lesion_threshold=config.get("small_lesion_threshold", 0.5),
                            patch_size=tuple(config.get("small_lesion_patch_size", (16, 16, 16))),
                            small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
                            # Parametry pro pokročilou kombinaci predikcí
                            alpha=config.get("combine_alpha", 0.6),
                            confidence_boost_factor=config.get("combine_boost_factor", 1.5),
                            high_conf_threshold=config.get("combine_high_conf_threshold", 0.8),
                            adaptive_weight=config.get("combine_adaptive", True),
                            size_based_weight=config.get("combine_size_based", True)
                        )
                        
                        # Výpis metrik
                        if result_item["metrics"]:
                            metrics = result_item["metrics"]
                            patient_id = result_item.get('patient_id', f'idx_{val_idx}')
                            print(f"    Pacient {patient_id}: Dice={metrics['dice']:.4f}, MASD={metrics['masd']:.4f}, NSD={metrics['nsd']:.4f}")
                        
                        # Uložení segmentace a vizualizací pomocí nové funkce pro 3 sloupce
                        from src.inference import save_validation_results_pdf
                        
                        # Upravíme prefix, aby odrážel, že jde o nejlepší model
                        prefix = f"best_val_{val_idx_pos}" if using_best_model else f"val_{val_idx_pos}"
                        
                        # Uložení MHA souboru a standardního PDF
                        save_segmentation_with_metrics(result_item, output_dir, prefix=prefix, save_pdf_comparison=False)
                        
                        # Uložení nového PDF se třemi sloupci (ZADC, LABEL, PRED)
                        pdf_prefix = f"best_val3col_{val_idx_pos}" if using_best_model else f"val3col_{val_idx_pos}"
                        save_validation_results_pdf(result_item, output_dir, prefix=pdf_prefix)
                    
                    # Vrácení modelu do původního stavu, pokud byl načten nejlepší model
                    if using_best_model:
                        print("  Obnovuji původní stav modelu z aktuální epochy")
                        model.load_state_dict(current_model_state)
                    
                    print(f"Validační inference dokončena. Výsledky jsou uloženy v: {output_dir}")
        
        # Ukládání metrik pro fold
        all_fold_metrics.append(fold_metrics)
        
        # Výpis výsledků pro tento fold
        if config.get("fixed_split", False):
            print(f"\nVýsledky pro PEVNÉ ROZDĚLENÍ:")
        else:
            print(f"\nVýsledky pro FOLD {fold_idx+1}:")
            
        for k, v in fold_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Výpis průměrných výsledků přes všechny foldy
    if not config.get("fixed_split", False) and len(all_fold_metrics) > 1:
        print("\n========== CELKOVÉ VÝSLEDKY ==========")
        avg_metrics = {}
        for metric in all_fold_metrics[0].keys():
            avg_value = sum(fold[metric] for fold in all_fold_metrics) / len(all_fold_metrics)
            avg_metrics[metric] = avg_value
            print(f"Průměr {metric}: {avg_value:.4f}")
        
        # Logování finálních metrik do wandb
        if config["use_wandb"]:
            for k, v in avg_metrics.items():
                wandb.run.summary[f"avg_{k}"] = v
    else:
        # Pevné rozdělení nebo pouze jeden fold
        avg_metrics = all_fold_metrics[0]
        
        if config["use_wandb"]:
            for k, v in avg_metrics.items():
                wandb.run.summary[k] = v
    
    # Ukončení wandb
    if config["use_wandb"]:
        wandb.finish()
    
    return avg_metrics


def run_inference(config):
    """
    Provede inferenci na testovacích datech.
    """
    print("\n========== INFERENCE ==========")
    
    # Kontrola, zda jsou všechny potřebné cesty zadány
    if not all(config.get(key) for key in ["adc_folder", "model_name", "output_dir"]):
        print("Chybí povinné parametry pro inferenci!")
        return
    
    # Pokud je zadán model pro každý fold, použijeme ensemble
    use_ensemble = False
    if config.get("use_ensemble", False) and config.get("n_folds", 0) > 1:
        # Kontrola, zda existují modely pro všechny foldy
        missing_models = False
        model_paths = []
        small_lesion_model_paths = []  # Přidaná podpora pro ensemble malých lézí
        
        for fold_idx in range(config["n_folds"]):
            model_path = os.path.join(config["model_dir"], f"best_model_fold{fold_idx+1}.pth")
            
            # Cesta k modelu pro malé léze
            if config.get("use_cascaded_approach", False):
                small_lesion_model_dir = os.path.join(config.get("model_dir", config["output_dir"]), "small_lesion")
                small_lesion_path = os.path.join(small_lesion_model_dir, f"best_small_lesion_model_fold{fold_idx+1}.pth")
                
                if os.path.exists(small_lesion_path):
                    small_lesion_model_paths.append(small_lesion_path)
                else:
                    print(f"Warning: Model pro malé léze pro fold {fold_idx+1} nenalezen v: {small_lesion_path}")
            
            if not os.path.exists(model_path):
                print(f"Warning: Model pro fold {fold_idx+1} nenalezen v: {model_path}")
                missing_models = True
            else:
                model_paths.append(model_path)
        
        if missing_models:
            print("Některé modely pro ensemble chybí. Použiji pouze model z prvního foldu.")
            use_ensemble = False
        else:
            print(f"Použiji ensemble {len(model_paths)} modelů.")
            if config.get("use_cascaded_approach", False) and small_lesion_model_paths:
                print(f"Použiji ensemble {len(small_lesion_model_paths)} modelů pro malé léze.")
            use_ensemble = True
    
    # Vytvoření adresáře pro výstupy
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Načtení seznamu souborů pro inferenci
    adc_files = sorted([f for f in os.listdir(config["adc_folder"]) if f.endswith(('.mha', '.nii', '.nii.gz'))])
    
    # Pokud máme Z-ADC, načteme i je
    if config["in_channels"] > 1:
        z_files = sorted([f for f in os.listdir(config["z_folder"]) if f.endswith(('.mha', '.nii', '.nii.gz'))])
        assert len(adc_files) == len(z_files), "Počet ADC a Z-ADC souborů nesouhlasí!"
    
    # Ground truth, pokud je k dispozici
    if config.get("label_folder", None):
        label_files = sorted([f for f in os.listdir(config["label_folder"]) if f.endswith(('.mha', '.nii', '.nii.gz'))])
        assert len(adc_files) == len(label_files), "Počet ADC a label souborů nesouhlasí!"
    else:
        label_files = [None] * len(adc_files)
    
    print(f"Celkem {len(adc_files)} souborů pro inferenci.")
    
    # Nastavení device
    device = torch.device(config["device"])
    
    # Vytvoření a přesunutí modelu na GPU
    if use_ensemble:
        # Pro ensemble vytvoříme seznam modelů
        models = []
        for model_path in model_paths:
            model = create_model(
                model_name=config["model_name"],
                pretrained=False,
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                img_size=config["img_size"],
                use_checkpointing=config.get("use_checkpointing", False),
                drop_rate=config.get("drop_rate", 0.0),
                attn_drop_rate=config.get("attn_drop_rate", 0.0),
                dropout_path_rate=config.get("dropout_path_rate", 0.0),
                apply_sigmoid=config.get("apply_sigmoid", False),
                swin_layers_per_block=config.get("swin_layers_per_block", None),
                swin_window_size=config.get("swin_window_size", None)
            ).to(device)
            
            # Načtení vah
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
        
        main_model = models[0]  # Pro kompatibilitu s kaskádovým přístupem
    else:
        # Pro jediný model
        main_model = create_model(
            model_name=config["model_name"],
            pretrained=False,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            img_size=config["img_size"],
            use_checkpointing=config.get("use_checkpointing", False),
            drop_rate=config.get("drop_rate", 0.0),
            attn_drop_rate=config.get("attn_drop_rate", 0.0),
            dropout_path_rate=config.get("dropout_path_rate", 0.0),
            apply_sigmoid=config.get("apply_sigmoid", False),
            swin_layers_per_block=config.get("swin_layers_per_block", None),
            swin_window_size=config.get("swin_window_size", None)
        ).to(device)
        
        # Načtení vah
        if config.get("model_weights", None):
            print(f"Načítám váhy modelu z: {config['model_weights']}")
            main_model.load_state_dict(torch.load(config["model_weights"], map_location=device))
        
        main_model.eval()
    
    if config.get("use_cascaded_approach", False):
        # Import funkcí pro malé léze
        from src.models import create_small_lesion_model
        from src.inference import infer_full_volume_cascaded
        
        # Kontrola, zda je cesta k modelu pro malé léze zadána nebo máme ensemble
        small_lesion_model = None
        small_lesion_models = []  # Pro ensemble
        
        if use_ensemble and small_lesion_model_paths:
            # Vytvoření ensemble modelů pro malé léze
            for sl_path in small_lesion_model_paths:
                sl_model = create_small_lesion_model(
                    model_name=config.get("small_lesion_model", "small_unet"),
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"]
                ).to(device)
                
                try:
                    sl_model.load_state_dict(torch.load(sl_path, map_location=device))
                    sl_model.eval()
                    small_lesion_models.append(sl_model)
                except Exception as e:
                    print(f"Chyba při načítání modelu pro malé léze {sl_path}: {e}")
            
            # Pokud máme alespoň jeden model, použijeme první jako referenční
            if small_lesion_models:
                print(f"Načteno {len(small_lesion_models)} modelů pro malé léze pro ensemble")
                small_lesion_model = small_lesion_models[0]  # Referenční model pro kompatibilitu
            else:
                print("Nepodařilo se načíst žádný model pro malé léze pro ensemble")
        elif not config.get("small_lesion_model_path", None):
            print("Varování: Kaskádový přístup vyžaduje model pro malé léze!")
            
            # Pokud máme k dispozici modely pro jednotlivé foldy, pokusíme se najít model pro první fold
            if config.get("use_ensemble", False) and config.get("n_folds", 0) > 1:
                # Zkusíme najít model pro první fold
                small_lesion_model_dir = os.path.join(config.get("model_dir", config["output_dir"]), "small_lesion")
                small_lesion_model_path = os.path.join(small_lesion_model_dir, "best_small_lesion_model_fold1.pth")
                
                if os.path.exists(small_lesion_model_path):
                    print(f"Nalezen model pro malé léze pro první fold: {small_lesion_model_path}")
                    config["small_lesion_model_path"] = small_lesion_model_path
                else:
                    print("Inicializuji nový model pro malé léze (bez předtrénovaných vah)...")
            else:
                print("Inicializuji nový model pro malé léze (bez předtrénovaných vah)...")
            
            # Vytvoření nového modelu pro malé léze, pokud stále nemáme cestu
            if not config.get("small_lesion_model_path", None):
                small_lesion_model = create_small_lesion_model(
                    model_name=config.get("small_lesion_model", "small_unet"),
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"]
                ).to(device)
            else:
                # Načtení existujícího modelu pro malé léze
                print(f"Načítám model pro malé léze z: {config['small_lesion_model_path']}")
                small_lesion_model = create_small_lesion_model(
                    model_name=config.get("small_lesion_model", "small_unet"),
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"]
                ).to(device)
                
                try:
                    small_lesion_model.load_state_dict(torch.load(config["small_lesion_model_path"], map_location=device))
                    small_lesion_model.eval()
                except Exception as e:
                    print(f"Chyba při načítání modelu pro malé léze: {e}")
                    print("Inicializuji nový model pro malé léze (bez předtrénovaných vah)...")
    
    # Inference na všech souborech
    for idx, (adc_file, z_file) in enumerate(zip(adc_files, z_files)):
        adc_path = os.path.join(config["adc_folder"], adc_file)
        z_path = os.path.join(config["z_folder"], z_file)
        
        # Kontrola, zda existuje odpovídající ground truth
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
        
        print(f"\nInference pro vzorek {idx+1}/{len(adc_files)}: {adc_file}")
        
        # Výběr metody inference
        if config.get("use_cascaded_approach", False) and small_lesion_model is not None:
            # Kaskádový přístup
            print("Použití kaskádového přístupu s modelem pro malé léze...")
            
            # Určení, který model pro malé léze použít
            if use_ensemble and small_lesion_models:
                # Provádíme inferenci pro každý model v ensemblu a průměrujeme výsledky
                print("Provádím ensemble inferenci s modely pro malé léze...")
                
                predictions = []
                
                for i, (main_m, small_m) in enumerate(zip(models, small_lesion_models)):
                    print(f"Ensemble model {i+1}/{len(models)}...")
                    
                    # Načtení vstupních dat
                    input_paths = [adc_path, z_path]
                    volumes = []
                    
                    # Vždy načíst ADC mapu (první v seznamu)
                    adc_path_curr = input_paths[0]
                    sitk_img = sitk.ReadImage(adc_path_curr)
                    np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                    volumes.append(np_vol)
                    
                    # Načíst Z-ADC mapu, pouze pokud se používá
                    if config["in_channels"] > 1 and len(input_paths) > 1:
                        zadc_path = input_paths[1]
                        try:
                            sitk_zadc = sitk.ReadImage(zadc_path)
                            zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
                            volumes.append(zadc_np)
                        except Exception as e:
                            print(f"Varování: Nelze načíst Z-ADC soubor {zadc_path}: {e}")
                    
                    # Vytvoření vstupního tensoru
                    input_vol = np.stack(volumes, axis=0)  # tvar: (C, D, H, W)
                    
                    # Získání transformací pro TTA, pokud je povoleno
                    tta_transforms = None
                    if config.get("use_tta", True):
                        tta_transforms = get_tta_transforms(angle_max=config.get("tta_angle_max", 3))
                    
                    # Inference s aktuálním ensemblem modelů
                    result_item = infer_full_volume_cascaded(
                        input_vol=input_vol,
                        main_model=main_m,
                        small_lesion_model=small_m,
                        device=device,
                        use_tta=config.get("use_tta", True),
                        tta_transforms=tta_transforms,
                        cascaded_mode=config.get("cascaded_mode", "combined"),
                        small_lesion_threshold=config.get("small_lesion_threshold", 0.5),
                        patch_size=tuple(config.get("small_lesion_patch_size", (16, 16, 16))),
                        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
                        # Parametry pro pokročilou kombinaci predikcí
                        alpha=config.get("combine_alpha", 0.6),
                        confidence_boost_factor=config.get("combine_boost_factor", 1.5),
                        high_conf_threshold=config.get("combine_high_conf_threshold", 0.8),
                        adaptive_weight=config.get("combine_adaptive", True),
                        size_based_weight=config.get("combine_size_based", True)
                    )
                    predictions.append(result_item["prediction"])
                
                # Průměrování predikcí ensemblu
                ensemble_pred = np.mean([pred for pred in predictions], axis=0).astype(np.int32)
                result = {
                    "prediction": ensemble_pred,  # Výsledek je průměr predikcí
                    "metrics": {}
                }
                
                # Pokud máme ground truth, spočítáme metriky
                if label_path:
                    lab_sitk = sitk.ReadImage(label_path)
                    lab_np = sitk.GetArrayFromImage(lab_sitk)
                    result["metrics"] = compute_all_metrics(ensemble_pred, lab_np, include_surface_metrics=True)
            else:
                # Standardní inference s jedním modelem pro malé léze
                input_paths = [adc_path, z_path]
                # Načtení vstupních dat
                volumes = []
                
                # Vždy načíst ADC mapu (první v seznamu)
                adc_path = input_paths[0]
                sitk_img = sitk.ReadImage(adc_path)
                np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                volumes.append(np_vol)
                
                # Načíst Z-ADC mapu, pouze pokud se používá
                if config["in_channels"] > 1 and len(input_paths) > 1:
                    zadc_path = input_paths[1]
                    try:
                        sitk_zadc = sitk.ReadImage(zadc_path)
                        zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
                        volumes.append(zadc_np)
                    except Exception as e:
                        print(f"Varování: Nelze načíst Z-ADC soubor {zadc_path}: {e}")
                        print("Inference bude provedena pouze s ADC mapou.")
                
                # Vytvoření vstupního tensoru
                input_vol = np.stack(volumes, axis=0)  # tvar: (C, D, H, W)
                
                # Získání transformací pro TTA, pokud je povoleno
                tta_transforms = None
                if config.get("use_tta", True):
                    tta_transforms = get_tta_transforms(angle_max=config.get("tta_angle_max", 3))
                
                result = infer_full_volume_cascaded(
                    input_vol=input_vol,
                    main_model=main_model,
                    small_lesion_model=small_lesion_model,
                    device=device,
                    use_tta=config.get("use_tta", True),
                    tta_transforms=tta_transforms,
                    cascaded_mode=config.get("cascaded_mode", "combined"),
                    small_lesion_threshold=config.get("small_lesion_threshold", 0.5),
                    patch_size=tuple(config.get("small_lesion_patch_size", (16, 16, 16))),
                    small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
                    # Parametry pro pokročilou kombinaci predikcí
                    alpha=config.get("combine_alpha", 0.6),
                    confidence_boost_factor=config.get("combine_boost_factor", 1.5),
                    high_conf_threshold=config.get("combine_high_conf_threshold", 0.8),
                    adaptive_weight=config.get("combine_adaptive", True),
                    size_based_weight=config.get("combine_size_based", True)
                )
        
        elif config["inference_mode"] == "moe" and expert_model is not None:
            # MoE inference
            result = infer_full_volume_moe(
                main_model=main_model,
                expert_model=expert_model,
                input_paths=[adc_path, z_path],
                label_path=label_path,
                device=device,
                threshold=config["moe_threshold"],
                use_z_adc=config["in_channels"] > 1
            )
            
            # Výpis informací o použitém modelu
            print(f"Použitý model: {result['model_used']}, foreground voxely: {result['foreground_voxels']}")
            
        else:
            # Standardní inference
            result = infer_full_volume(
                model=main_model,
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
        
        # Výpis metrik, pokud jsou k dispozici
        if label_path and result["metrics"]:
            metrics = result["metrics"]
            print(f"Metriky: Dice={metrics['dice']:.4f}, MASD={metrics['masd']:.4f}, NSD={metrics['nsd']:.4f}")
        
        # Uložení výsledků
        output_dir = config["output_dir"]
        save_segmentation_with_metrics(result, output_dir, save_pdf_comparison=config["save_pdf_comparison"])
    
    print(f"\nInference dokončena. Výsledky jsou uloženy v: {config['output_dir']}")


def main():
    """Hlavní funkce programu"""
    # Získání výchozí konfigurace
    config = get_default_config()
    
    # Parsování argumentů
    parser = argparse.ArgumentParser(description="SWINUNetR pro segmentaci HIE lézí")
    
    # Režim běhu
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Režim běhu (train nebo inference)")
    
    # Argumenty datasetu
    parser.add_argument("--adc_folder", type=str, help="Cesta ke složce s ADC snímky")
    parser.add_argument("--z_folder", type=str, help="Cesta ke složce s Z-ADC snímky")
    parser.add_argument("--label_folder", type=str, help="Cesta ke složce s ground truth maskami")
    parser.add_argument("--preprocessed_adc_folder", type=str, help="Cesta k předzpracované ADC složce")
    parser.add_argument("--preprocessed_z_folder", type=str, help="Cesta k předzpracované Z_ADC složce")
    parser.add_argument("--preprocessed_label_folder", type=str, help="Cesta k předzpracované LABEL složce")
    parser.add_argument("--extended_dataset", action="store_true", 
                        help="Použít rozšířený dataset (s aug/orig soubory)")
    parser.add_argument("--max_aug_per_orig", type=int, default=0,
                        help="Maximální počet augmentovaných souborů na jeden originální")
    
    # Argumenty modelu
    parser.add_argument("--model_name", type=str, help="Jméno modelu")
    parser.add_argument("--model_path", type=str, help="Cesta k uloženému modelu pro inference")
    parser.add_argument("--expert_model_path", type=str, help="Cesta k expertnímu modelu pro MoE inference")
    parser.add_argument("--small_lesion_model_path", type=str, help="Cesta k modelu pro detekci malých lézí")
    parser.add_argument("--in_channels", type=int, help="Počet vstupních kanálů")
    parser.add_argument("--out_channels", type=int, help="Počet výstupních tříd")
    parser.add_argument("--drop_rate", type=float, help="Dropout rate")
    
    # Argumenty trénování
    parser.add_argument("--batch_size", type=int, help="Velikost dávky")
    parser.add_argument("--epochs", type=int, help="Počet epoch")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--eta_min", type=float, help="Minimální learning rate pro scheduler")
    parser.add_argument("--training_mode", type=str, choices=["patch", "full_volume"], 
                        help="Režim trénování (patch nebo full_volume)")
    parser.add_argument("--no_augmentation", action="store_false", dest="use_augmentation",
                        help="Vypnout augmentaci dat")
    parser.add_argument("--augmentation_type", type=str, choices=["soft", "heavy"], default="soft",
                        help="Typ augmentace pro trénování ('soft' pro lehké augmentace, 'heavy' pro silnější augmentace)")
    parser.add_argument("--n_folds", type=int, help="Počet foldů pro cross-validaci")
    parser.add_argument("--use_ohem", action="store_true", help="Povolit Online Hard Example Mining")
    parser.add_argument("--ohem_ratio", type=float, default=0.15, help="Poměr těžkých příkladů pro OHEM")
    parser.add_argument("--ohem_start_epoch", type=int, default=1, help="Epocha, od které se začne používat OHEM")
    parser.add_argument("--patches_per_volume", type=int, help="Počet patchů na objem při patch-based trénování")
    parser.add_argument("--patch_size", type=int, nargs=3, help="Velikost patche (3 hodnoty: výška, šířka, hloubka)")
    parser.add_argument("--inference_every_n_epochs", type=int, default=0, help="Provést inferenci každých N epoch (0 = vypnuto)")
    
    # Přidané argumenty pro patch-based training
    parser.add_argument("--intelligent_sampling", action="store_true", help="Povolit inteligentní vzorkování patchů zaměřené na léze")
    parser.add_argument("--foreground_ratio", type=float, default=0.7, help="Poměr patchů, které by měly obsahovat léze (0-1)")
    parser.add_argument("--sw_overlap", type=float, default=0.5, help="Míra překrytí pro sliding window inference (0-1)")
    
    # Argumenty inference
    parser.add_argument("--inference_mode", type=str, choices=["standard", "moe"], default="standard", 
                        help="Režim inference (standard nebo moe)")
    parser.add_argument("--no_tta", action="store_false", dest="use_tta",
                        help="Vypnout Test-Time Augmentation při inferenci")
    parser.add_argument("--moe_threshold", type=int, help="Threshold pro přepnutí na expertní model")
    parser.add_argument("--save_pdf_comparison", action="store_true",
                        help="Vytvořit PDF s porovnáním ground truth a predikce (jen když je dostupný label)")
    
    # Obecné argumenty
    parser.add_argument("--seed", type=int, help="Seed pro reprodukovatelnost")
    parser.add_argument("--device", type=str, help="Zařízení pro výpočet (cuda nebo cpu)")
    parser.add_argument("--output_dir", type=str, help="Výstupní adresář")
    
    # Parametry dat a normalizace
    parser.add_argument("--use_normalization", action="store_true", 
                        help="Použít normalizaci dat")
    parser.add_argument("--allow_normalize_spacing", action="store_true", 
                        help="Povolit normalizaci rozestupů voxelů")
    parser.add_argument("--preprocessing", action="store_true",
                        help="Provést preprocessing dat (bounding box, crop, padding) před trénováním")
    
    # Parametry fixního rozdělení dat
    parser.add_argument("--fixed_split", action="store_true",
                        help="Použít pevné rozdělení dat místo k-fold CV (80/20)")
    parser.add_argument("--fixed_split_seed", type=int, default=None,
                        help="Seed pro vytvoření pevného rozdělení dat (pokud se liší od hlavního seedu)")
    parser.add_argument("--fixed_split_file", type=str, default=None,
                        help="Cesta k souboru s uloženým pevným rozdělením dat (.pkl)")

    # Parametry ztráty a metrik
    parser.add_argument("--loss_name", type=str, 
                        choices=["weighted_ce_dice", "log_cosh_dice", "focal_tversky", 
                                 "log_hausdorff", "focal", "focal_dice_combo", 
                                 "focal_ce_combo", "dice_focal", "weighted_ce"],
                        help="Jméno ztrátové funkce")
    parser.add_argument("--focal_alpha", type=float, help="Alpha parametr pro Focal loss")
    parser.add_argument("--focal_gamma", type=float, help="Gamma parametr pro Focal loss")
    parser.add_argument("--alpha_mix", type=float, help="Míchací parametr pro ztrátové funkce")
    parser.add_argument("--compute_surface_metrics", action="store_true", 
                        help="Počítat povrchové metriky (MASD, NSD)")

    # TTA parametry
    parser.add_argument("--tta_angle_max", type=int, help="Maximální úhel pro TTA rotace")

    # Wandb argumenty
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb",
                        help="Vypnout logování do wandb")
    parser.add_argument("--wandb_project", type=str, help="Jméno projektu ve wandb")
    parser.add_argument("--wandb_run_name", type=str, help="Jméno běhu ve wandb")
    
    # Přidané argumenty pro kaskádový přístup
    parser.add_argument("--use_cascaded_approach", action="store_true", 
                        help="Použít kaskádový přístup pro segmentaci malých lézí")
    parser.add_argument("--cascaded_mode", type=str, choices=["roi_only", "combined"], default="combined",
                        help="Režim kaskádového přístupu: 'roi_only' jen přidá ROI jako kanál, 'combined' kombinuje předpovědi")
    parser.add_argument("--small_lesion_model", type=str, choices=["unet", "nnunet", "deeplabv3plus", "small_unet", "simple_resunet", "attention_unet"],
                        default="small_unet", help="Model pro detekci malých lézí")
    parser.add_argument("--small_lesion_patch_size", nargs=3, type=int, default=[16, 16, 16],
                        help="Velikost patche pro model malých lézí")
    parser.add_argument("--small_lesion_threshold", type=float, default=0.5,
                        help="Práh pro detekci malých lézí")
    parser.add_argument("--small_lesion_epochs", type=int, default=50,
                        help="Počet epoch pro trénování modelu malých lézí")
    parser.add_argument("--small_lesion_batch_size", type=int, default=16,
                        help="Velikost dávky pro trénování modelu malých lézí")
    parser.add_argument("--small_lesion_lr", type=float, default=0.001,
                        help="Learning rate pro trénování modelu malých lézí")
    parser.add_argument("--small_lesion_patches_per_volume", type=int, default=200,
                        help="Počet patchů na objem pro trénování modelu malých lézí")
    parser.add_argument("--small_lesion_foreground_ratio", type=float, default=0.8,
                        help="Poměr foreground voxelů v trénovacích patchích pro model malých lézí")
    parser.add_argument("--small_lesion_max_voxels", type=int, default=50,
                        help="Maximální počet voxelů pro klasifikaci léze jako 'malé'")
    parser.add_argument("--small_lesion_large_lesion_sampling_ratio", type=float, default=0.25,
                        help="Poměr redukce vzorků z velkých lézí (0-1, výchozí 0.25)")
    parser.add_argument("--small_lesion_loss_name", type=str, default="focal_ce_combo",
                        help="Ztrátová funkce pro trénování modelu malých lézí")

    # Parametry pro pokročilou kombinaci predikcí
    parser.add_argument("--advanced_combine", action="store_true", 
                        help="Použít pokročilou kombinaci predikcí")
    parser.add_argument("--combine_alpha", type=float, default=0.6,
                        help="Základní váha hlavního modelu při kombinaci (0-1)")
    parser.add_argument("--combine_boost_factor", type=float, default=1.5,
                        help="Faktor zvýšení váhy pro detekce s vysokou jistotou")
    parser.add_argument("--combine_high_conf_threshold", type=float, default=0.8,
                        help="Práh pravděpodobnosti pro klasifikaci jako 'vysoká jistota'")
    parser.add_argument("--combine_disable_adaptive", action="store_false", dest="combine_adaptive",
                        help="Vypnout adaptivní váhování podle velikosti a jistoty")
    parser.add_argument("--combine_disable_size_weighting", action="store_false", dest="combine_size_based",
                        help="Vypnout váhování podle velikosti léze")

    # Aktualizace konfigurace z argumentů
    args = parser.parse_args()
    config = parse_args_to_config(args, config)
    
    # Spuštění požadovaného režimu
    if config["mode"] == "train":
        print("Spouštění trénování...")
        run_cross_validation(config)
    else:
        print("Spouštění inference...")
        run_inference(config)


if __name__ == "__main__":
    main() 