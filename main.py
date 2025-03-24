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
    train_with_ohem
)
from src.inference.inference import (
    infer_full_volume,
    infer_full_volume_moe,
    save_segmentation_with_metrics
)

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
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
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
                    
                    # Zpracujeme všechny validační vzorky, ne jen první
                    for val_idx_pos, val_idx in enumerate(val_indices):
                        print(f"  Vzorek {val_idx_pos+1}/{len(val_indices)} (index {val_idx})")
                        
                        # Cesty k datům
                        adc_path = os.path.join(config["adc_folder"], full_dataset.adc_files[val_idx])
                        z_path = os.path.join(config["z_folder"], full_dataset.z_files[val_idx])
                        label_path = os.path.join(config["label_folder"], full_dataset.lab_files[val_idx])
                        
                        # Provedení inference
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
                        
                        # Výpis metrik
                        if result["metrics"]:
                            metrics = result["metrics"]
                            patient_id = result.get('patient_id', f'idx_{val_idx}')
                            print(f"    Pacient {patient_id}: Dice={metrics['dice']:.4f}, MASD={metrics['masd']:.4f}, NSD={metrics['nsd']:.4f}")
                        
                        # Uložení segmentace a vizualizací pomocí nové funkce pro 3 sloupce
                        from src.inference import save_validation_results_pdf
                        
                        # Uložení MHA souboru a standardního PDF
                        save_segmentation_with_metrics(result, output_dir, prefix=f"val_{val_idx_pos}", save_pdf_comparison=False)
                        
                        # Uložení nového PDF se třemi sloupci (ZADC, LABEL, PRED)
                        save_validation_results_pdf(result, output_dir, prefix=f"val3col_{val_idx_pos}")
                    
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
    Spustí inferenci na zadaných datech.
    
    Args:
        config: Konfigurační slovník s parametry
    """
    # Kontrola, zda složky existují
    for folder in [config["adc_folder"], config["z_folder"]]:
        if not os.path.exists(folder):
            print(f"Chyba: Složka {folder} neexistuje!")
            sys.exit(1)
    
    # Provedení preprocessingu, pokud je požadováno
    if config.get("preprocessing", False):
        from src.data.preprocessing import prepare_preprocessed_data
        
        # Nastavení cest pro předzpracovaná data
        preprocessed_adc = config.get("preprocessed_adc_folder", os.path.join(config["output_dir"], "preprocessed/1ADC_ss"))
        preprocessed_z = config.get("preprocessed_z_folder", os.path.join(config["output_dir"], "preprocessed/2Z_ADC"))
        preprocessed_label = None
        if config.get("label_folder"):
            preprocessed_label = config.get("preprocessed_label_folder", os.path.join(config["output_dir"], "preprocessed/3LABEL"))
        
        # Spuštění preprocessingu
        print("\nProvádím preprocessing dat...")
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
        
        # Aktualizace cest na předzpracovaná data
        config["adc_folder"] = preprocessed_adc
        config["z_folder"] = preprocessed_z
        if config.get("label_folder") and preprocessed_label:
            config["label_folder"] = preprocessed_label
        
        print("Preprocessing dokončen. Používám předzpracovaná data pro inferenci.\n")
    
    # Kontrola, zda existuje cesta k modelu
    if not os.path.exists(config["model_path"]):
        print(f"Chyba: Model {config['model_path']} neexistuje!")
        sys.exit(1)
    
    # Načtení seznamu souborů
    adc_files = sorted([f for f in os.listdir(config["adc_folder"]) 
                        if f.endswith(".mha") or f.endswith(".nii") or f.endswith(".nii.gz")])
    z_files = sorted([f for f in os.listdir(config["z_folder"]) 
                      if f.endswith(".mha") or f.endswith(".nii") or f.endswith(".nii.gz")])
    
    if len(adc_files) != len(z_files):
        print("Chyba: Počet ADC a Z-ADC souborů se neshoduje!")
        sys.exit(1)
    
    # Vytvoření výstupního adresáře
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Určení device
    device = torch.device(config["device"])
    
    # Vytvoření a načtení modelu
    from src.models import create_model
    
    model = create_model(
        model_name=config["model_name"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        drop_rate=config.get("drop_rate", 0.15)
    ).to(device)
    
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()
    
    # Načtení expertního modelu, pokud se jedná o MoE inferenci
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
            
            # Výpis informací o použitém modelu
            print(f"Použitý model: {result['model_used']}, foreground voxely: {result['foreground_voxels']}")
            
        else:
            # Standardní inference
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