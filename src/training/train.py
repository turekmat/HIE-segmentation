import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import wandb
from monai.inferers import sliding_window_inference
import os
import time

from ..models import create_model
from ..loss import dice_coefficient
from ..data.preprocessing import get_tta_transforms
from ..data.dataset import get_subject_id_from_filename


def custom_collate_fn(batch):
    """
    Vlastní collate funkce, která umožňuje zpracovat tensory různých velikostí.
    Každý vzorek je zpracován samostatně, bez stack operace.
    
    Args:
        batch: List vzorků z datasetu
        
    Returns:
        List obsahující dvojice (inputs, labels)
    """
    return batch


def train_one_epoch(model, loader, optimizer, loss_fn, device="cuda"):
    """
    Trénuje model jednu epochu.
    
    Args:
        model: Model k trénování
        loader: DataLoader s trénovacími daty
        optimizer: Optimalizátor
        loss_fn: Ztrátová funkce
        device: Zařízení pro výpočet
        
    Returns:
        float: Průměrná ztráta za celou epochu
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
    Vylepšená verze sliding window inference s pokročilým vážením predikcí.
    
    Args:
        model: Model pro inferenci
        inputs: Vstupní tensor (B, C, D, H, W)
        roi_size: Velikost patche pro inferenci (D, H, W)
        sw_batch_size: Batch size pro sliding window
        overlap: Míra překrytí mezi sousedními patchy (0-1)
        mode: Metoda vážení překrývajících se predikcí ("gaussian" nebo "constant")
        device: Zařízení pro výpočet
        
    Returns:
        torch.Tensor: Predikce pro celý vstup
    """
    # Přidána kontrola zařízení
    if device is None:
        device = inputs.device
    
    # Přidána optimalizace vážení predikcí
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
    
    # Získání rozměrů vstupních dat a patch
    input_shape = list(inputs.shape[2:])  # [D, H, W]
    patch_shape = roi_size
    
    # Zjištění, které dimenze patche jsou stejně velké nebo větší než odpovídající dimenze vstupu
    # Pro tyto dimenze použijeme velmi malý overlap, abychom zabránili problémům s paddingem
    adaptive_overlap = [overlap] * len(input_shape)
    for i in range(len(input_shape)):
        if patch_shape[i] >= input_shape[i]:
            # Pro dimenze, kde je patch větší nebo stejný jako vstup, použijeme minimální overlap
            adaptive_overlap[i] = 0.01
            # Úprava velikosti patche, pokud je větší než vstup
            patch_shape[i] = min(patch_shape[i], input_shape[i])
    
    # Použití adaptivního overlapu
    try:
        # Využijeme MONAI implementaci
        return sliding_window_inference(
            inputs=inputs,
            roi_size=patch_shape,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=adaptive_overlap,
            **extra_params
        )
    except RuntimeError as e:
        # Záložní řešení v případě problémů - zkusíme použít menší overlap
        print(f"Varování: Chyba při sliding window inference. Zkouším s menším overlapem. Původní chyba: {e}")
        try:
            return sliding_window_inference(
                inputs=inputs,
                roi_size=patch_shape,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.01,  # Minimální overlap
                **extra_params
            )
        except Exception as e2:
            # Jako poslední možnost zkusíme zpracovat celý vstup najednou
            print(f"Varování: Záložní řešení selhalo. Pokusím se zpracovat vstup najednou. Chyba: {e2}")
            if inputs.shape[2] * inputs.shape[3] * inputs.shape[4] > 128 * 128 * 64:
                print("Varování: Vstup je velmi velký pro zpracování najednou!")
            return model(inputs)


def validate_one_epoch(model, loader, loss_fn, device="cuda", training_mode="full_volume",
                       compute_surface_metrics=True, USE_TTA=True, TTA_ANGLE_MAX=3,
                       batch_size=1, patch_size=(64, 64, 64), tta_forward_fn=None,
                       sw_overlap=0.5):
    """
    Validuje model na validační sadě.
    
    Args:
        model: Model k validaci
        loader: DataLoader s validačními daty
        loss_fn: Ztrátová funkce
        device: Zařízení pro výpočet
        training_mode: "full_volume" nebo "patch"
        compute_surface_metrics: Zda počítat metriky povrchu
        USE_TTA: Zda použít Test-Time Augmentation
        TTA_ANGLE_MAX: Maximální úhel pro TTA
        batch_size: Velikost dávky
        patch_size: Velikost patche pro inferenci
        tta_forward_fn: Funkce pro Test-Time Augmentation
        sw_overlap: Míra překrytí pro sliding window (0-1)
        
    Returns:
        dict: Slovník s metrikami
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_masd = 0.0
    running_nsd  = 0.0
    count_samples = 0
    
    # Kontrola velikosti patche pro SR data
    original_patch_size = patch_size
    
    tta_transforms = get_tta_transforms(angle_max=TTA_ANGLE_MAX) if USE_TTA else None

    with torch.no_grad():
        # Upraveno pro zpracování custom_collate_fn výstupu
        for batch in loader:
            for sample in batch:
                try:
                    inputs, labels = sample
                    inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
                    
                    # Pro super-resolution data kontrola rozměrů a potenciální úprava patch size
                    if training_mode == "patch":
                        input_shape = inputs.shape[2:]  # [D, H, W]
                        if any(p >= s for p, s in zip(patch_size, input_shape)):
                            # Vytvoříme nový patch size, který bude respektovat velikost vstupu
                            adjusted_patch_size = [min(p, s) for p, s in zip(patch_size, input_shape)]
                            patch_size = adjusted_patch_size
                    
                    if training_mode == "patch":
                        try:
                            # Použít optimalizovaný sliding window
                            logits = optimal_sliding_window_inference(
                                model=model, 
                                inputs=inputs, 
                                roi_size=patch_size, 
                                sw_batch_size=1, 
                                overlap=sw_overlap,
                                mode="gaussian",  # Nebo "constant" pro jednoduché průměrování
                                device=device
                            )
                        except Exception as e:
                            print(f"Chyba při sliding window inference: {e}")
                            print("Zkouším zpracovat vstup najednou...")
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
                            print(f"Chyba při TTA: {e}. Používám inferenci bez TTA.")
                            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                    else:
                        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

                    label = labels.cpu().numpy()[0]
                    dsc = dice_coefficient(pred, label)
                    running_dice += dsc

                    if compute_surface_metrics:
                        try:
                            # Zde by bylo dobré importovat compute_masd a compute_nsd ze správného místa
                            # například z utils.metrics
                            from ..utils.metrics import compute_masd, compute_nsd
                            masd = compute_masd(pred, label, spacing=(1,1,1), sampling_ratio=0.5)
                            nsd  = compute_nsd(pred, label, spacing=(1,1,1), sampling_ratio=0.5)
                            running_masd += masd
                            running_nsd  += nsd
                        except Exception as e:
                            print(f"Chyba při výpočtu povrchových metrik: {e}")
                            if count_samples == 0:  # Jen při prvním vzorku vypisovat detaily
                                print(f"Tvary: pred {pred.shape}, label {label.shape}")
                                
                    count_samples += 1
                except Exception as e:
                    print(f"Chyba při zpracování vzorku: {e}")
                    continue
    
    # Obnovení původní velikosti patch
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
    Připraví vše potřebné pro trénování (model, optimizer, loss_fn, scheduler).
    
    Args:
        config: Konfigurační slovník
        dataset_train: Trénovací dataset
        dataset_val: Validační dataset
        device: Zařízení pro výpočet
        
    Returns:
        tuple: (model, optimizer, loss_fn, scheduler, train_loader, val_loader)
    """
    # Vytvoření dataloaderů
    train_loader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
    
    # Pro validační loader používáme custom_collate_fn pro zpracování různých velikostí tensorů
    val_loader = DataLoader(
        dataset_val, 
        batch_size=config.get("val_batch_size", 2),  # Můžeme použít menší batch size pro validaci
        shuffle=False,
        collate_fn=custom_collate_fn  # Použití vlastní collate_fn
    )
    
    # Vytvoření modelu
    model = create_model(
        model_name=config["model_name"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        drop_rate=config.get("drop_rate", 0.15)
    ).to(device)
    
    # Vytvoření ztrátové funkce
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
    
    # Vytvoření optimizeru a scheduleru
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["eta_min"]
    )
    
    return model, optimizer, loss_fn, scheduler, train_loader, val_loader


def train_with_ohem(model, dataset_train, optimizer, loss_fn, batch_size=1, device="cuda", ohem_ratio=0.15):
    """
    Trénuje model pomocí Online Hard Example Mining (OHEM).
    OHEM vybírá patche, které jsou pro model nejtěžší (mají nejvyšší ztrátu).
    
    Args:
        model: Model k trénování
        dataset_train: Dataset s trénovacími daty
        optimizer: Optimalizátor
        loss_fn: Ztrátová funkce
        batch_size: Velikost dávky
        device: Zařízení pro výpočet
        ohem_ratio: Poměr těžkých příkladů k vybranému počtu (např. 0.15 = 15% nejtěžších patchů)
        
    Returns:
        float: Průměrná ztráta za epochu
    """
    model.train()
    max_samples = 5000  # Maximální počet vzorků pro OHEM analýzu
    sample_limit = min(len(dataset_train), max_samples)
    weighted_subset_size = min(int(sample_limit * ohem_ratio), batch_size * 100)  # Omezení počtu vzorků
    
    print(f"OHEM: Analýza {sample_limit} vzorků z celkových {len(dataset_train)}")
    
    loss_per_sample = []
    indices = []
    
    # Uvolnění paměti
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Získáme ztrátu pro každý vzorek
    for idx in range(sample_limit):
        try:
            inputs, labels = dataset_train[idx]
            inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                loss_per_sample.append(loss.item())
                indices.append(idx)
                
            # Uvolnění paměti
            del inputs, labels, logits, loss
            if idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Chyba při OHEM analýze vzorku {idx}: {e}")
            continue
    
    if not loss_per_sample:
        print("Varování: Žádné vzorky nebyly úspěšně analyzovány pro OHEM. Přeskakuji OHEM trénink.")
        return 0.0
    
    # Seřadíme vzorky podle ztráty (sestupně) a vybereme nejhorší
    sorted_indices = [indices[i] for i in np.argsort(loss_per_sample)[::-1]]
    hard_indices = sorted_indices[:weighted_subset_size]
    
    print(f"OHEM: Vybráno {len(hard_indices)} těžkých vzorků pro trénink")
    
    # Vytvoříme Subset datasetu s těžkými vzorky
    from torch.utils.data import Subset, DataLoader
    hard_subset = Subset(dataset_train, hard_indices)
    weighted_train_loader = DataLoader(hard_subset, batch_size=batch_size, shuffle=True)
    
    # Trénování na těžkých patchech
    running_loss = 0.0
    batch_count = 0
    
    # Nastavíme model do trénovacího módu
    model.train()
    
    # Uvolnění paměti
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Trénování na těžkých vzorcích s gradient accumulation pro stabilitu
    for inputs, labels in weighted_train_loader:
        try:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Uvolnění paměti
            del inputs, labels, outputs, loss
            if batch_count % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Chyba při OHEM tréninku na dávce: {e}")
            continue
    
    # Vrátíme průměrnou ztrátu
    return running_loss / max(1, batch_count)


def create_cv_folds(dataset, n_folds, extended_dataset=False):
    """
    Rozdělí dataset na k-fold cross-validační sady.
    
    Args:
        dataset: Dataset k rozdělení
        n_folds: Počet foldů
        extended_dataset: Zda se jedná o rozšířený dataset s orig/aug soubory
        
    Returns:
        list: Seznam dvojic (train_indices, val_indices)
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
    
    # Seznam všech subjektů (promíchaný)
    all_subjects = list(subject_to_indices.keys())
    np.random.shuffle(all_subjects)
    
    # Výpočet velikosti foldu
    num_subjects = len(all_subjects)
    fold_size = num_subjects // n_folds
    
    print(f"Celkem {num_subjects} unikátních subjektů, vytváříme {n_folds}-fold cross-validaci.\n")
    
    folds = []
    for fold_idx in range(n_folds):
        # Výběr validačních subjektů
        fold_val_start = fold_idx * fold_size
        fold_val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else num_subjects
        val_subjects = all_subjects[fold_val_start:fold_val_end]
        train_subjects = list(set(all_subjects) - set(val_subjects))
        
        # Výběr validačních indexů
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
        
        print(f"Fold {fold_idx+1}: Val subjekty={len(val_subjects)}, Val vzorky={len(val_indices)}, Trén subjekty={len(train_subjects)}, Trén vzorky={len(train_indices)}")
        folds.append((train_indices, val_indices))
    
    return folds


def log_metrics(metrics, epoch, fold_idx=None, lr=None, wandb_enabled=True):
    """
    Loguje metriky do konzole a do wandb.
    
    Args:
        metrics: Slovník s metrikami
        epoch: Aktuální epocha
        fold_idx: Index aktuálního foldu (None pro pevné rozdělení)
        lr: Aktuální learning rate
        wandb_enabled: Zda je wandb povoleno
    """
    # Výpis do konzole
    log_str = ""
    if fold_idx is not None:
        log_str += f"[FOLD {fold_idx+1}] "
    else:
        log_str += "[PEVNÉ ROZDĚLENÍ] "
    
    log_str += f"Epoch {epoch} => "
    
    for k, v in metrics.items():
        log_str += f"{k}={v:.4f}, "
    
    if lr is not None:
        log_str += f"lr={lr:.6f}"
    
    print(log_str)
    
    # Logování do wandb
    if wandb_enabled:
        wandb_log = metrics.copy()
        
        if fold_idx is not None:
            wandb_log['fold'] = fold_idx + 1
        else:
            wandb_log['fold'] = 0  # 0 označuje pevné rozdělení
        
        wandb_log['epoch'] = epoch
        
        if lr is not None:
            wandb_log['lr'] = lr
        
        wandb.log(wandb_log)


def setup_small_lesion_training(config, dataset_train, dataset_val, device="cuda"):
    """
    Nastaví model pro trénování malých lézí a vrátí potřebné komponenty.
    
    Args:
        config: Konfigurační slovník
        dataset_train: Trénovací dataset
        dataset_val: Validační dataset
        device: Zařízení pro výpočet
    
    Returns:
        tuple: (model, optimizer, loss_fn, scheduler, train_loader, val_loader)
    """
    from ..models import create_small_lesion_model
    
    # Vytvoření dataloaderů
    train_loader = DataLoader(
        dataset_train, 
        batch_size=config.get("small_lesion_batch_size", 16),
        shuffle=True
    )
    
    # Pro validační loader používáme custom_collate_fn pro zpracování různých velikostí tensorů
    val_loader = DataLoader(
        dataset_val, 
        batch_size=config.get("small_lesion_val_batch_size", 8),  
        shuffle=False,
        collate_fn=custom_collate_fn  # Použití vlastní collate_fn
    )
    
    # Vytvoření modelu
    model = create_small_lesion_model(
        model_name=config.get("small_lesion_model", "small_unet"),
        in_channels=config["in_channels"],
        out_channels=config["out_channels"]
    ).to(device)
    
    # Vytvoření ztrátové funkce
    from ..loss import get_loss_function
    
    class_weights = None
    if config["loss_name"] == "weighted_ce_dice":
        class_weights = torch.tensor(
            [config["bg_weight"], config["fg_weight"]],
            device=device,
            dtype=torch.float
        )
    
    loss_fn = get_loss_function(
        loss_name=config.get("small_lesion_loss_name", config["loss_name"]),
        alpha=config["alpha"],
        class_weights=class_weights,
        focal_alpha=config.get("focal_alpha", 0.75),
        focal_gamma=config.get("focal_gamma", 2.0),
        alpha_mix=config.get("alpha_mix", 0.6),
        out_channels=config["out_channels"]
    )
    
    # Vytvoření optimizeru a scheduleru
    small_lesion_lr = config.get("small_lesion_lr", config["lr"])
    optimizer = optim.Adam(model.parameters(), lr=small_lesion_lr)
    
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("small_lesion_epochs", config["epochs"]),
        eta_min=config["eta_min"]
    )
    
    return model, optimizer, loss_fn, scheduler, train_loader, val_loader


def train_small_lesion_model(config, device="cuda"):
    """
    Trénuje model pro detekci malých lézí.
    
    Args:
        config: Konfigurační slovník
        device: Zařízení pro výpočet
        
    Returns:
        str: Cesta k natrénovanému modelu
    """
    print("\n========== TRÉNOVÁNÍ MODELU PRO DETEKCI MALÝCH LÉZÍ ==========")
    
    # Import potřebných funkcí
    from ..data.dataset import SmallLesionPatchDataset
    
    # Vytvoření trénovacího a validačního datasetu
    small_patch_size = config.get("small_lesion_patch_size", (16, 16, 16))
    
    # Kontrola, zda je velikost patche ve správném formátu
    if isinstance(small_patch_size, list):
        small_patch_size = tuple(small_patch_size)
    
    print(f"Vytváření datasetu malých lézí s velikostí patche {small_patch_size}")
    
    # Vytvoření datasetu pro trénink a validaci
    # Používáme celý dataset pro trénink, ale s malými patchi
    train_dataset = SmallLesionPatchDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        patch_size=small_patch_size,
        patches_per_volume=config.get("small_lesion_patches_per_volume", 200),
        foreground_ratio=config.get("small_lesion_foreground_ratio", 0.8),
        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
        augment=config.get("use_augmentation", True),
        use_z_adc=config["in_channels"] > 1,
        seed=config["seed"],
        large_lesion_sampling_ratio=config.get("small_lesion_large_lesion_sampling_ratio", 0.25)
    )
    
    # PŘIDANÉ: Diagnostické statistiky o datasetu malých lézí
    print("\n===== STATISTIKY MALÝCH LÉZÍ V DATASETU =====")
    
    # Analýza distribuce lézí v datasetu
    small_lesion_count = 0
    larger_lesion_count = 0
    no_lesion_count = 0
    total_lesion_voxels = 0
    
    for info in train_dataset.volume_info:
        if info['is_small_lesion'] and info['has_lesions']:
            small_lesion_count += 1
            total_lesion_voxels += info['lesion_voxels']
        elif info['has_lesions']:
            larger_lesion_count += 1
        else:
            no_lesion_count += 1
    
    print(f"Objemy s malými lézemi: {small_lesion_count}")
    print(f"Objemy s většími lézemi: {larger_lesion_count}")
    print(f"Objemy bez lézí: {no_lesion_count}")
    print(f"Celkem objemů: {len(train_dataset.volume_info)}")
    
    # Histogramy velikostí lézí
    lesion_sizes = [info['lesion_voxels'] for info in train_dataset.volume_info if info['has_lesions']]
    
    if lesion_sizes:
        min_size = min(lesion_sizes)
        max_size = max(lesion_sizes)
        avg_size = sum(lesion_sizes) / len(lesion_sizes)
        
        print(f"Statistika velikostí lézí:")
        print(f"  Minimální: {min_size} voxelů")
        print(f"  Maximální: {max_size} voxelů")
        print(f"  Průměrná: {avg_size:.1f} voxelů")
        print(f"  Práh pro malé léze: {config.get('small_lesion_max_voxels', 50)} voxelů")
        
        # Kategorie velikostí
        size_categories = {
            "velmi malé (≤20 voxelů)": 0,
            "malé (21-50 voxelů)": 0,
            "střední (51-200 voxelů)": 0,
            "velké (>200 voxelů)": 0
        }
        
        for size in lesion_sizes:
            if size <= 20:
                size_categories["velmi malé (≤20 voxelů)"] += 1
            elif size <= 50:
                size_categories["malé (21-50 voxelů)"] += 1
            elif size <= 200:
                size_categories["střední (51-200 voxelů)"] += 1
            else:
                size_categories["velké (>200 voxelů)"] += 1
        
        print("Distribuce velikostí lézí:")
        for category, count in size_categories.items():
            print(f"  {category}: {count} objemů ({(count/len(lesion_sizes))*100:.1f}%)")
    
    print(f"\nCelkem voxelů v malých lézích: {total_lesion_voxels}")
    print(f"Průměrná velikost malé léze: {total_lesion_voxels / max(1, small_lesion_count):.1f} voxelů")
    
    # Pro validaci oddělíme 20% dat z trénovacího datasetu, ale zachováme stejnou strategii vzorkování
    print("Vytváření validační množiny se stejnou strategií vzorkování...")
    
    # Rozdělíme objemy na trénovací a validační
    volume_info = train_dataset.volume_info
    np.random.seed(config["seed"])
    np.random.shuffle(volume_info)
    
    dataset_size = len(volume_info)
    split = int(np.floor(0.2 * dataset_size))
    
    # Vytvoříme seznamy souborů pro validaci
    val_volume_indices = [info['index'] for info in volume_info[:split]]
    val_adc_files = [train_dataset.adc_files[i] for i in val_volume_indices]
    val_label_files = [train_dataset.lab_files[i] for i in val_volume_indices]
    
    if config["in_channels"] > 1:
        val_z_files = [train_dataset.z_files[i] for i in val_volume_indices]
    else:
        val_z_files = None
    
    # Vytvoření validačního datasetu se stejnou strategií vzorkování
    val_dataset = SmallLesionPatchDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        patch_size=small_patch_size,
        patches_per_volume=config.get("small_lesion_patches_per_volume", 200),
        foreground_ratio=config.get("small_lesion_foreground_ratio", 0.8),
        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
        augment=False,  # Bez augmentace pro validaci
        use_z_adc=config["in_channels"] > 1,
        seed=config["seed"] + 1,  # Jiný seed pro validační data
        specific_files={
            'adc_files': val_adc_files,
            'z_files': val_z_files,
            'lab_files': val_label_files
        },
        large_lesion_sampling_ratio=config.get("small_lesion_large_lesion_sampling_ratio", 0.25)
    )
    
    # Aktualizace trénovacího datasetu bez validačních vzorků
    train_indices = [i for i in range(len(train_dataset.adc_files)) if i not in val_volume_indices]
    
    # Vytvoření upraveného trénovacího datasetu
    updated_train_adc_files = [train_dataset.adc_files[i] for i in train_indices]
    updated_train_label_files = [train_dataset.lab_files[i] for i in train_indices]
    
    if config["in_channels"] > 1:
        updated_train_z_files = [train_dataset.z_files[i] for i in train_indices]
    else:
        updated_train_z_files = None
    
    # Vytvoření nového trénovacího datasetu
    train_dataset = SmallLesionPatchDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        patch_size=small_patch_size,
        patches_per_volume=config.get("small_lesion_patches_per_volume", 200),
        foreground_ratio=config.get("small_lesion_foreground_ratio", 0.8),
        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
        augment=config.get("use_augmentation", True),
        use_z_adc=config["in_channels"] > 1,
        seed=config["seed"],
        specific_files={
            'adc_files': updated_train_adc_files,
            'z_files': updated_train_z_files,
            'lab_files': updated_train_label_files
        },
        large_lesion_sampling_ratio=config.get("small_lesion_large_lesion_sampling_ratio", 0.25)
    )
    
    print(f"Velikost validačního datasetu: {len(val_dataset)} vzorků")
    print(f"Velikost upraveného trénovacího datasetu: {len(train_dataset)} vzorků")
    
    # Inicializace modelu a tréninkových komponent
    model, optimizer, loss_fn, scheduler, train_loader, val_loader = setup_small_lesion_training(
        config=config,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        device=device
    )
    
    # PŘIDANÉ: Výpis parametrů modelu a optimizace
    print("\n===== PARAMETRY MODELU A TRÉNOVÁNÍ =====")
    print(f"Model: {config.get('small_lesion_model', 'small_unet')}")
    print(f"Vstupní kanály: {config['in_channels']}")
    print(f"Výstupní třídy: {config['out_channels']}")
    print(f"Ztrátová funkce: {config.get('small_lesion_loss_name', config['loss_name'])}")
    print(f"Learning rate: {config.get('small_lesion_lr', config['lr'])}")
    
    # Počet parametrů modelu
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Počet parametrů modelu: {total_params:,}")
    
    # Počet trénovacích epoch
    epochs = config.get("small_lesion_epochs", config["epochs"])
    print(f"Počet trénovacích epoch: {epochs}")
    
    # Ukládání nejlepšího modelu
    best_val_dice = 0.0
    model_dir = config.get("small_lesion_model_dir", config["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "best_small_lesion_model.pth")
    
    # Tréninkový cyklus
    print("\n===== PRŮBĚH TRÉNOVÁNÍ =====")
    
    # Sledování metrik během trénování
    best_epoch = 0
    epoch_times = []
    train_losses = []
    val_dices = []
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Trénování jedné epochy
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
            training_mode="patch",  # Vždy patch-based pro malé léze
            compute_surface_metrics=config["compute_surface_metrics"],
            USE_TTA=False,  # Pro zrychlení vypneme TTA při validaci
            patch_size=small_patch_size,
            sw_overlap=config.get("sw_overlap", 0.5)
        )
        
        # Aktualizace learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Logování metrik
        metrics = {
            'small_lesion_train_loss': train_loss,
            **{f"small_lesion_{k}": v for k, v in val_metrics.items()}
        }
        
        log_metrics(
            metrics=metrics,
            epoch=epoch,
            fold_idx=None,
            lr=curr_lr,
            wandb_enabled=config["use_wandb"]
        )
        
        # Sledování metrik
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        val_dices.append(val_metrics.get('val_dice', 0.0))
        
        # Výpis detailních metrik
        dice_metric = val_metrics.get('val_dice', 0.0)
        masd_metric = val_metrics.get('val_masd', float('inf'))
        nsd_metric = val_metrics.get('val_nsd', 0.0)
        
        print(f"Epocha {epoch}/{epochs}: ", end="")
        print(f"Ztráta = {train_loss:.4f}, DICE = {dice_metric:.4f}", end="")
        
        if 'val_masd' in val_metrics:
            print(f", MASD = {masd_metric:.4f}", end="")
        
        if 'val_nsd' in val_metrics:
            print(f", NSD = {nsd_metric:.4f}", end="")
            
        print(f" [čas: {epoch_time:.1f}s]")
        
        # Ukládání nejlepšího modelu
        if dice_metric > best_val_dice:
            improvement = dice_metric - best_val_dice
            best_val_dice = dice_metric
            best_epoch = epoch
            
            # Uložení modelu
            print(f"Nalezen lepší model (DICE = {best_val_dice:.4f}, zlepšení: +{improvement:.4f}), ukládám...")
            torch.save(model.state_dict(), model_save_path)
    
    # PŘIDÁNO: Souhrn trénování
    print("\n===== SOUHRN TRÉNOVÁNÍ MODELU MALÝCH LÉZÍ =====")
    print(f"Nejlepší model (epocha {best_epoch}): DICE = {best_val_dice:.4f}")
    print(f"Průměrná doba epochy: {sum(epoch_times)/len(epoch_times):.1f}s")
    print(f"Celková doba trénování: {sum(epoch_times)/60:.1f} minut")
    
    # Grafy vývoje metrik (textová reprezentace)
    print("\nVývoj ztrátové funkce:")
    for i, loss in enumerate(train_losses, 1):
        print(f"Epocha {i}: {'#' * int(loss * 20)}")
    
    print("\nVývoj DICE koeficientu:")
    for i, dice in enumerate(val_dices, 1):
        print(f"Epocha {i}: {'#' * int(dice * 40)}")
    
    print("\nTrénování modelu pro detekci malých lézí dokončeno.")
    print(f"Nejlepší model uložen v: {model_save_path}")
    
    return model_save_path


def train_small_lesion_model_with_indices(config, train_indices, fold_idx=None, device="cuda"):
    """
    Trénuje model pro detekci malých lézí pouze na datech určených indexy.
    Tato funkce je navržena pro použití v rámci cross-validace, kdy chceme
    pro každý fold natrénovat vlastní model pro malé léze pouze na trénovacích datech.
    
    Args:
        config: Konfigurační slovník
        train_indices: Seznam indexů pro trénování
        fold_idx: Index foldu (None pro bez cross-validace)
        device: Zařízení pro výpočet
        
    Returns:
        str: Cesta k natrénovanému modelu
    """
    fold_text = f" (FOLD {fold_idx+1})" if fold_idx is not None else ""
    print(f"\n========== TRÉNOVÁNÍ MODELU PRO DETEKCI MALÝCH LÉZÍ{fold_text} ==========")
    
    # Import potřebných funkcí
    from ..data.dataset import SmallLesionPatchDataset, BONBID3DFullVolumeDataset
    from torch.utils.data import Subset
    
    # Vytvoření trénovacího a validačního datasetu
    small_patch_size = config.get("small_lesion_patch_size", (16, 16, 16))
    
    # Kontrola, zda je velikost patche ve správném formátu
    if isinstance(small_patch_size, list):
        small_patch_size = tuple(small_patch_size)
    
    print(f"Vytváření datasetu malých lézí s velikostí patche {small_patch_size}")
    
    # Nejprve vytvoříme full dataset a poté vybereme pouze trénovací indexy
    full_dataset = BONBID3DFullVolumeDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        augment=False,
        extended_dataset=config.get("extended_dataset", False),
        max_aug_per_orig=config.get("max_aug_per_orig", 0),
        use_z_adc=config["in_channels"] > 1
    )
    
    # Výběr pouze trénovacích souborů podle indexů
    train_adc_files = [full_dataset.adc_files[i] for i in train_indices]
    train_label_files = [full_dataset.lab_files[i] for i in train_indices]
    
    if config["in_channels"] > 1:
        train_z_files = [full_dataset.z_files[i] for i in train_indices]
    else:
        train_z_files = None
    
    print(f"Vybráno {len(train_adc_files)} souborů pro trénink modelu malých lézí")
    
    # Vytvoření datasetu pro trénink a validaci
    # Používáme pouze vybrané soubory pro trénink
    train_dataset = SmallLesionPatchDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        patch_size=small_patch_size,
        patches_per_volume=config.get("small_lesion_patches_per_volume", 200),
        foreground_ratio=config.get("small_lesion_foreground_ratio", 0.8),
        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
        augment=config.get("use_augmentation", True),
        use_z_adc=config["in_channels"] > 1,
        seed=config["seed"],
        specific_files={
            'adc_files': train_adc_files,
            'z_files': train_z_files,
            'lab_files': train_label_files
        },
        large_lesion_sampling_ratio=config.get("small_lesion_large_lesion_sampling_ratio", 0.25)
    )
    
    # PŘIDANÉ: Diagnostické statistiky o datasetu malých lézí
    print("\n===== STATISTIKY MALÝCH LÉZÍ V DATASETU =====")
    
    # Analýza distribuce lézí v datasetu
    small_lesion_count = 0
    larger_lesion_count = 0
    no_lesion_count = 0
    total_lesion_voxels = 0
    
    for info in train_dataset.volume_info:
        if info['is_small_lesion'] and info['has_lesions']:
            small_lesion_count += 1
            total_lesion_voxels += info['lesion_voxels']
        elif info['has_lesions']:
            larger_lesion_count += 1
        else:
            no_lesion_count += 1
    
    print(f"Objemy s malými lézemi: {small_lesion_count}")
    print(f"Objemy s většími lézemi: {larger_lesion_count}")
    print(f"Objemy bez lézí: {no_lesion_count}")
    print(f"Celkem objemů: {len(train_dataset.volume_info)}")
    
    # Histogramy velikostí lézí
    lesion_sizes = [info['lesion_voxels'] for info in train_dataset.volume_info if info['has_lesions']]
    
    if lesion_sizes:
        min_size = min(lesion_sizes)
        max_size = max(lesion_sizes)
        avg_size = sum(lesion_sizes) / len(lesion_sizes)
        
        print(f"Statistika velikostí lézí:")
        print(f"  Minimální: {min_size} voxelů")
        print(f"  Maximální: {max_size} voxelů")
        print(f"  Průměrná: {avg_size:.1f} voxelů")
        print(f"  Práh pro malé léze: {config.get('small_lesion_max_voxels', 50)} voxelů")
        
        # Kategorie velikostí
        size_categories = {
            "velmi malé (≤20 voxelů)": 0,
            "malé (21-50 voxelů)": 0,
            "střední (51-200 voxelů)": 0,
            "velké (>200 voxelů)": 0
        }
        
        for size in lesion_sizes:
            if size <= 20:
                size_categories["velmi malé (≤20 voxelů)"] += 1
            elif size <= 50:
                size_categories["malé (21-50 voxelů)"] += 1
            elif size <= 200:
                size_categories["střední (51-200 voxelů)"] += 1
            else:
                size_categories["velké (>200 voxelů)"] += 1
        
        print("Distribuce velikostí lézí:")
        for category, count in size_categories.items():
            print(f"  {category}: {count} objemů ({(count/len(lesion_sizes))*100:.1f}%)")
    
    print(f"\nCelkem voxelů v malých lézích: {total_lesion_voxels}")
    print(f"Průměrná velikost malé léze: {total_lesion_voxels / max(1, small_lesion_count):.1f} voxelů")
    
    # Pro validaci oddělíme 20% dat z trénovacího datasetu, ale zachováme stejnou strategii vzorkování
    print("Vytváření validační množiny se stejnou strategií vzorkování...")
    
    # Rozdělíme objemy na trénovací a validační
    volume_info = train_dataset.volume_info
    np.random.seed(config["seed"])
    np.random.shuffle(volume_info)
    
    dataset_size = len(volume_info)
    split = int(np.floor(0.2 * dataset_size))
    
    # Vytvoříme seznamy souborů pro validaci
    val_volume_indices = [info['index'] for info in volume_info[:split]]
    val_adc_files = [train_dataset.adc_files[i] for i in val_volume_indices]
    val_label_files = [train_dataset.lab_files[i] for i in val_volume_indices]
    
    if config["in_channels"] > 1:
        val_z_files = [train_dataset.z_files[i] for i in val_volume_indices]
    else:
        val_z_files = None
    
    # Vytvoření validačního datasetu se stejnou strategií vzorkování
    val_dataset = SmallLesionPatchDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        patch_size=small_patch_size,
        patches_per_volume=config.get("small_lesion_patches_per_volume", 200),
        foreground_ratio=config.get("small_lesion_foreground_ratio", 0.8),
        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
        augment=False,  # Bez augmentace pro validaci
        use_z_adc=config["in_channels"] > 1,
        seed=config["seed"] + 1,  # Jiný seed pro validační data
        specific_files={
            'adc_files': val_adc_files,
            'z_files': val_z_files,
            'lab_files': val_label_files
        },
        large_lesion_sampling_ratio=config.get("small_lesion_large_lesion_sampling_ratio", 0.25)
    )
    
    # Aktualizace trénovacího datasetu bez validačních vzorků
    train_indices = [i for i in range(len(train_dataset.adc_files)) if i not in val_volume_indices]
    
    # Vytvoření upraveného trénovacího datasetu
    updated_train_adc_files = [train_dataset.adc_files[i] for i in train_indices]
    updated_train_label_files = [train_dataset.lab_files[i] for i in train_indices]
    
    if config["in_channels"] > 1:
        updated_train_z_files = [train_dataset.z_files[i] for i in train_indices]
    else:
        updated_train_z_files = None
    
    # Vytvoření nového trénovacího datasetu
    train_dataset = SmallLesionPatchDataset(
        adc_folder=config["adc_folder"],
        z_folder=config["z_folder"],
        label_folder=config["label_folder"],
        patch_size=small_patch_size,
        patches_per_volume=config.get("small_lesion_patches_per_volume", 200),
        foreground_ratio=config.get("small_lesion_foreground_ratio", 0.8),
        small_lesion_max_voxels=config.get("small_lesion_max_voxels", 50),
        augment=config.get("use_augmentation", True),
        use_z_adc=config["in_channels"] > 1,
        seed=config["seed"],
        specific_files={
            'adc_files': updated_train_adc_files,
            'z_files': updated_train_z_files,
            'lab_files': updated_train_label_files
        },
        large_lesion_sampling_ratio=config.get("small_lesion_large_lesion_sampling_ratio", 0.25)
    )
    
    print(f"Velikost validačního datasetu: {len(val_dataset)} vzorků")
    print(f"Velikost upraveného trénovacího datasetu: {len(train_dataset)} vzorků")
    
    # Inicializace modelu a tréninkových komponent
    model, optimizer, loss_fn, scheduler, train_loader, val_loader = setup_small_lesion_training(
        config=config,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        device=device
    )
    
    # PŘIDANÉ: Výpis parametrů modelu a optimizace
    print("\n===== PARAMETRY MODELU A TRÉNOVÁNÍ =====")
    print(f"Model: {config.get('small_lesion_model', 'small_unet')}")
    print(f"Vstupní kanály: {config['in_channels']}")
    print(f"Výstupní třídy: {config['out_channels']}")
    print(f"Ztrátová funkce: {config.get('small_lesion_loss_name', config['loss_name'])}")
    print(f"Learning rate: {config.get('small_lesion_lr', config['lr'])}")
    
    # Počet parametrů modelu
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Počet parametrů modelu: {total_params:,}")
    
    # Počet trénovacích epoch
    epochs = config.get("small_lesion_epochs", config["epochs"])
    print(f"Počet trénovacích epoch: {epochs}")
    
    # Ukládání nejlepšího modelu
    best_val_dice = 0.0
    model_dir = config.get("small_lesion_model_dir", os.path.join(config["model_dir"], "small_lesion"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Název souboru modelu s využitím indexu foldu
    model_filename = f"best_small_lesion_model_fold{fold_idx+1}.pth" if fold_idx is not None else "best_small_lesion_model.pth"
    model_save_path = os.path.join(model_dir, model_filename)
    
    # Tréninkový cyklus
    print("\n===== PRŮBĚH TRÉNOVÁNÍ =====")
    
    # Sledování metrik během trénování
    best_epoch = 0
    epoch_times = []
    train_losses = []
    val_dices = []
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Trénování jedné epochy
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
            training_mode="patch",  # Vždy patch-based pro malé léze
            compute_surface_metrics=config["compute_surface_metrics"],
            USE_TTA=False,  # Pro zrychlení vypneme TTA při validaci
            patch_size=small_patch_size,
            sw_overlap=config.get("sw_overlap", 0.5)
        )
        
        # Aktualizace learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Logování metrik
        metrics = {
            f"small_lesion_fold{fold_idx+1}_train_loss" if fold_idx is not None else 'small_lesion_train_loss': train_loss,
            **{f"small_lesion_fold{fold_idx+1}_{k}" if fold_idx is not None else f"small_lesion_{k}": v for k, v in val_metrics.items()}
        }
        
        log_metrics(
            metrics=metrics,
            epoch=epoch,
            fold_idx=fold_idx,
            lr=curr_lr,
            wandb_enabled=config["use_wandb"]
        )
        
        # Sledování metrik
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        val_dices.append(val_metrics.get('val_dice', 0.0))
        
        # Výpis detailních metrik
        dice_metric = val_metrics.get('val_dice', 0.0)
        masd_metric = val_metrics.get('val_masd', float('inf'))
        nsd_metric = val_metrics.get('val_nsd', 0.0)
        
        fold_info = f"[Fold {fold_idx+1}] " if fold_idx is not None else ""
        print(f"{fold_info}Epocha {epoch}/{epochs}: ", end="")
        print(f"Ztráta = {train_loss:.4f}, DICE = {dice_metric:.4f}", end="")
        
        if 'val_masd' in val_metrics:
            print(f", MASD = {masd_metric:.4f}", end="")
        
        if 'val_nsd' in val_metrics:
            print(f", NSD = {nsd_metric:.4f}", end="")
            
        print(f" [čas: {epoch_time:.1f}s]")
        
        # Ukládání nejlepšího modelu
        if dice_metric > best_val_dice:
            improvement = dice_metric - best_val_dice
            best_val_dice = dice_metric
            best_epoch = epoch
            
            # Uložení modelu
            print(f"{fold_info}Nalezen lepší model (DICE = {best_val_dice:.4f}, zlepšení: +{improvement:.4f}), ukládám...")
            torch.save(model.state_dict(), model_save_path)
    
    # PŘIDÁNO: Souhrn trénování
    print(f"\n===== SOUHRN TRÉNOVÁNÍ MODELU MALÝCH LÉZÍ{fold_text} =====")
    print(f"Nejlepší model (epocha {best_epoch}): DICE = {best_val_dice:.4f}")
    print(f"Průměrná doba epochy: {sum(epoch_times)/len(epoch_times):.1f}s")
    print(f"Celková doba trénování: {sum(epoch_times)/60:.1f} minut")
    
    # Grafy vývoje metrik (textová reprezentace)
    print("\nVývoj ztrátové funkce:")
    for i, loss in enumerate(train_losses, 1):
        print(f"Epocha {i}: {'#' * int(loss * 20)}")
    
    # Vrácení cesty k nejlepšímu modelu
    return model_save_path 