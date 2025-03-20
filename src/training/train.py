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


def validate_one_epoch(model, loader, loss_fn, device="cuda", training_mode="full_volume",
                       compute_surface_metrics=True, USE_TTA=True, TTA_ANGLE_MAX=3,
                       batch_size=1, patch_size=(64, 64, 64), tta_forward_fn=None):
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
        
    Returns:
        dict: Slovník s metrikami
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_masd = 0.0
    running_nsd  = 0.0
    count_samples = 0

    tta_transforms = get_tta_transforms(angle_max=TTA_ANGLE_MAX) if USE_TTA else None

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if training_mode == "patch":
                logits = sliding_window_inference(inputs, patch_size, batch_size, model, overlap=0.25)
            else:
                logits = model(inputs)

            loss = loss_fn(logits, labels)
            running_loss += loss.item()

            if USE_TTA and tta_forward_fn is not None:
                batch_preds = []
                for i in range(inputs.shape[0]):
                    input_i = inputs[i].unsqueeze(0)
                    avg_probs = tta_forward_fn(model, input_i, device, tta_transforms)
                    pred_i = np.argmax(avg_probs, axis=0)
                    batch_preds.append(pred_i)
                preds = np.array(batch_preds)
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            labs = labels.cpu().numpy()
            for pred, label in zip(preds, labs):
                dsc = dice_coefficient(pred, label)
                running_dice += dsc

                if compute_surface_metrics:
                    # Zde by bylo dobré importovat compute_masd a compute_nsd ze správného místa
                    # například z utils.metrics
                    from ..utils.metrics import compute_masd, compute_nsd
                    masd = compute_masd(pred, label, spacing=(1,1,1), sampling_ratio=0.5)
                    nsd  = compute_nsd(pred, label, spacing=(1,1,1), sampling_ratio=0.5)
                    running_masd += masd
                    running_nsd  += nsd
                count_samples += 1

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / count_samples if count_samples > 0 else 0.0

    metrics = {'val_loss': avg_loss, 'val_dice': avg_dice}
    if compute_surface_metrics and count_samples > 0:
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
    val_loader = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False)
    
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
    weighted_subset_size = min(int(len(dataset_train) * ohem_ratio), batch_size * 100)  # Omezení počtu vzorků
    loss_per_sample = []
    indices = []
    
    # Získáme ztrátu pro každý vzorek
    for idx in range(min(len(dataset_train), 5000)):  # Omezení na 5000 vzorků pro rychlost
        inputs, labels = dataset_train[idx]
        inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss_per_sample.append(loss.item())
            indices.append(idx)
    
    # Seřadíme vzorky podle ztráty (sestupně) a vybereme nejhorší
    sorted_indices = [indices[i] for i in np.argsort(loss_per_sample)[::-1]]
    hard_indices = sorted_indices[:weighted_subset_size]
    
    # Vytvoříme Subset datasetu s těžkými vzorky
    from torch.utils.data import Subset, DataLoader
    hard_subset = Subset(dataset_train, hard_indices)
    weighted_train_loader = DataLoader(hard_subset, batch_size=batch_size, shuffle=True)
    
    # Trénování na těžkých patchech
    running_loss = 0.0
    batch_count = 0
    
    # Nastavíme model do trénovacího módu
    model.train()
    
    # Trénování na těžkých vzorcích s gradient accumulation pro stabilitu
    for inputs, labels in weighted_train_loader:
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
        fold_idx: Index aktuálního foldu
        lr: Aktuální learning rate
        wandb_enabled: Zda je wandb povoleno
    """
    # Výpis do konzole
    log_str = ""
    if fold_idx is not None:
        log_str += f"[FOLD {fold_idx+1}] "
    
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
        
        wandb_log['epoch'] = epoch
        
        if lr is not None:
            wandb_log['lr'] = lr
        
        wandb.log(wandb_log) 