import os
import torch
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
import torch.nn.functional as F

from ..utils.metrics import dice_coefficient, compute_masd, compute_nsd
from ..data.preprocessing import get_tta_transforms, apply_tta_transform, invert_tta_transform
from ..data.dataset import extract_patient_id
from .feature_extraction import extract_swinunetr_features

def tta_forward(model, input_tensor, device, tta_transforms):
    """
    Provede inference s Test-Time Augmentation pro jeden vstupní vzorek.

    Args:
        model: Natrénovaný model
        input_tensor: Torch tensor se tvarem (1, C, D, H, W)
        device: Cílové zařízení ("cuda" nebo "cpu")
        tta_transforms: Seznam transformací (získáno pomocí get_tta_transforms)

    Returns:
        np.ndarray: Průměrná pravděpodobnostní mapa (tvar: (num_classes, D, H, W))
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=1)

    input_np = input_tensor.cpu().numpy()[0]  # tvar: (C, D, H, W)
    accumulated_probs = None

    for transform in tta_transforms:
        aug_vol = apply_tta_transform(input_np, transform)
        aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
        
        with torch.no_grad():
            logits = model(aug_tensor)
        
        probs = softmax(logits)
        probs_np = probs.cpu().numpy()[0]
        inv_probs = invert_tta_transform(probs_np, transform)

        if accumulated_probs is None:
            accumulated_probs = inv_probs
        else:
            accumulated_probs += inv_probs

    avg_probs = accumulated_probs / len(tta_transforms)
    return avg_probs


def infer_full_volume(model, 
                     input_paths, 
                     label_path=None, 
                     device="cuda", 
                     use_tta=True, 
                     tta_angle_max=3,
                     training_mode="full_volume",
                     patch_size=(64, 64, 64),
                     batch_size=1,
                     use_z_adc=True):
    """
    Provede inferenci pro celý 3D objem.

    Args:
        model: Natrénovaný model
        input_paths: Seznam cest ke vstupním volumům (např. [adc_path, zadc_path])
        label_path: Cesta k ground truth masce (volitelné)
        device: Zařízení pro výpočet
        use_tta: Zda použít Test-Time Augmentation
        tta_angle_max: Maximální úhel pro rotace při TTA
        training_mode: "full_volume" nebo "patch"
        patch_size: Velikost patche pro patch-based inferenci
        batch_size: Velikost dávky pro patch-based inferenci
        use_z_adc: Zda používat Z-ADC modalitu (druhý vstupní kanál)

    Returns:
        dict: Výsledky inference, včetně predikce a metrik (pokud je k dispozici ground truth)
    """
    model.eval()
    tta_transforms = get_tta_transforms(angle_max=tta_angle_max) if use_tta else None
    
    # Načtení vstupních dat
    volumes = []
    
    # Vždy načíst ADC mapu (první v seznamu)
    adc_path = input_paths[0]
    sitk_img = sitk.ReadImage(adc_path)
    np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    volumes.append(np_vol)
    
    # Načíst Z-ADC mapu, pouze pokud se používá
    if use_z_adc and len(input_paths) > 1:
        zadc_path = input_paths[1]
        try:
            sitk_zadc = sitk.ReadImage(zadc_path)
            zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
            volumes.append(zadc_np)
        except Exception as e:
            print(f"Varování: Nelze načíst Z-ADC soubor {zadc_path}: {e}")
            print("Inference bude provedena pouze s ADC mapou.")
            use_z_adc = False  # Vypnutí Z-ADC, pokud soubor nelze načíst
    
    # Vytvoření vstupního tensoru
    input_vol = np.stack(volumes, axis=0)  # tvar: (C, D, H, W)
    input_tensor = torch.from_numpy(input_vol).unsqueeze(0).to(device).float()
    
    # Ground truth data (pokud jsou k dispozici)
    lab_np = None
    if label_path:
        lab_sitk = sitk.ReadImage(label_path)
        lab_np = sitk.GetArrayFromImage(lab_sitk)
    
    # Inference
    with torch.no_grad():
        if use_tta:
            avg_probs = 0
            for transform in tta_transforms:
                aug_vol = apply_tta_transform(input_vol, transform)
                aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
                
                if training_mode == "patch":
                    pred_logits = sliding_window_inference(
                        aug_tensor, patch_size, batch_size, model, overlap=0.25)
                else:
                    pred_logits = model(aug_tensor)
                
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(pred_logits)
                probs_np = probs.cpu().numpy()[0]
                inv_probs = invert_tta_transform(probs_np, transform)
                avg_probs += inv_probs
            
            avg_probs /= len(tta_transforms)
            pred_np = np.argmax(avg_probs, axis=0)
        else:
            if training_mode == "patch":
                pred_logits = sliding_window_inference(
                    input_tensor, patch_size, batch_size, model, overlap=0.25)
            else:
                pred_logits = model(input_tensor)
            
            pred_np = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
    
    # Výpočet metrik (pokud je k dispozici ground truth)
    metrics = {}
    if lab_np is not None:
        metrics["dice"] = dice_coefficient(pred_np, lab_np)
        metrics["masd"] = compute_masd(pred_np, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
        metrics["nsd"] = compute_nsd(pred_np, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
        
    result = {
        'prediction': pred_np,
        'reference': lab_np,
        'input_paths': input_paths,
        'label_path': label_path,
        'metrics': metrics
    }
    
    if label_path:
        patient_id = extract_patient_id(label_path)
        result['patient_id'] = patient_id
    
    return result


def infer_full_volume_moe(main_model, expert_model, input_paths, label_path=None, 
                         device="cuda", threshold=80, use_z_adc=True):
    """
    Provede inferenci s použitím Mixture of Experts přístupu.

    Args:
        main_model: Hlavní model
        expert_model: Expertní model pro malé léze
        input_paths: Seznam cest ke vstupním volumům
        label_path: Cesta k ground truth masce (volitelné)
        device: Zařízení pro výpočet
        threshold: Threshold pro přepnutí na expertní model
        use_z_adc: Zda používat Z-ADC modalitu (druhý vstupní kanál)

    Returns:
        dict: Výsledky inference, včetně predikce a metrik
    """
    main_model.eval()
    expert_model.eval()
    
    # Načtení vstupních dat
    volumes = []
    
    # Vždy načíst ADC mapu (první v seznamu)
    adc_path = input_paths[0]
    sitk_img = sitk.ReadImage(adc_path)
    np_vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    volumes.append(np_vol)
    
    # Načíst Z-ADC mapu, pouze pokud se používá
    if use_z_adc and len(input_paths) > 1:
        zadc_path = input_paths[1]
        try:
            sitk_zadc = sitk.ReadImage(zadc_path)
            zadc_np = sitk.GetArrayFromImage(sitk_zadc).astype(np.float32)
            volumes.append(zadc_np)
        except Exception as e:
            print(f"Varování: Nelze načíst Z-ADC soubor {zadc_path}: {e}")
            print("Inference bude provedena pouze s ADC mapou.")
            use_z_adc = False  # Vypnutí Z-ADC, pokud soubor nelze načíst
    
    # Vytvoření vstupního tensoru
    input_vol = np.stack(volumes, axis=0)  # tvar: (C, D, H, W)
    input_tensor = torch.from_numpy(input_vol).unsqueeze(0).to(device).float()
    
    # Ground truth data (pokud jsou k dispozici)
    lab_np = None
    if label_path:
        lab_sitk = sitk.ReadImage(label_path)
        lab_np = sitk.GetArrayFromImage(lab_sitk)
    
    # Inference s hlavním modelem
    with torch.no_grad():
        logits_main = main_model(input_tensor)
        pred_main = torch.argmax(logits_main, dim=1).cpu().numpy()[0]
    
    # Zjištění počtu foreground voxelů
    fg_count = np.sum(pred_main == 1)
    
    # Výběr modelu na základě počtu foreground voxelů
    if fg_count < threshold and fg_count > 1:
        with torch.no_grad():
            logits_expert = expert_model(input_tensor)
            pred_final = torch.argmax(logits_expert, dim=1).cpu().numpy()[0]
        model_used = 'expert'
    else:
        pred_final = pred_main
        model_used = 'main'
    
    # Výpočet metrik (pokud je k dispozici ground truth)
    metrics = {}
    if lab_np is not None:
        metrics["dice"] = dice_coefficient(pred_final, lab_np)
        metrics["masd"] = compute_masd(pred_final, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
        metrics["nsd"] = compute_nsd(pred_final, lab_np, spacing=(1,1,1), sampling_ratio=0.5)
    
    result = {
        'prediction': pred_final,
        'reference': lab_np,
        'input_paths': input_paths,
        'label_path': label_path,
        'metrics': metrics,
        'model_used': model_used,
        'foreground_voxels': fg_count
    }
    
    if label_path:
        patient_id = extract_patient_id(label_path)
        result['patient_id'] = patient_id
    
    return result


def save_segmentation_to_file(prediction, reference_sitk, output_path):
    """
    Uloží predikovanou segmentaci do souboru.

    Args:
        prediction: Predikovaná segmentace (numpy array)
        reference_sitk: Referenční SimpleITK obraz pro získání metadat
        output_path: Cesta pro uložení segmentace
    """
    # Vytvoření SimpleITK obrazu z numpy array
    prediction = prediction.astype(np.uint8)
    output_sitk = sitk.GetImageFromArray(prediction)
    
    # Kopírování metadat z referenčního obrazu
    output_sitk.SetSpacing(reference_sitk.GetSpacing())
    output_sitk.SetOrigin(reference_sitk.GetOrigin())
    output_sitk.SetDirection(reference_sitk.GetDirection())
    
    # Vytvoření adresáře, pokud neexistuje
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Uložení obrazu
    sitk.WriteImage(output_sitk, output_path)
    print(f"Segmentace uložena do: {output_path}")


def save_slice_comparison_pdf(result, output_dir, prefix="comparison"):
    """
    Vytvoří PDF soubor s porovnáním všech řezů ground truth a predikce modelu.
    PDF bude obsahovat dva sloupce - v jednom sloupci ground truth a v druhém predikce.
    
    Args:
        result: Výsledek z infer_full_volume nebo infer_full_volume_moe
        output_dir: Výstupní adresář
        prefix: Prefix pro název souboru
        
    Returns:
        str: Cesta k vytvořenému PDF souboru
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction = result['prediction']  # shape (D, H, W)
    reference = result['reference']    # shape (D, H, W)
    
    if reference is None:
        print("Ground truth reference není k dispozici, nelze vytvořit PDF porovnání.")
        return None
    
    # Získání metadat pro název souboru
    metrics = result.get('metrics', {})
    dice = metrics.get('dice', 0.0)
    patient_id = result.get('patient_id', 'unknown')
    
    # Vytvoření cesty k souboru
    output_name = f"{prefix}_{patient_id}_dice{dice:.3f}.pdf"
    output_path = os.path.join(output_dir, output_name)
    
    # Zjištění počtu řezů a definice barev pro vizualizaci
    num_slices = reference.shape[0]
    rows_per_page = 6  # Počet řezů na stránku
    
    # Definice barevných map pro segmentace (pozadí průhledné, léze červená)
    gt_cmap = plt.cm.colors.ListedColormap(['none', 'red'])
    pred_cmap = plt.cm.colors.ListedColormap(['none', 'blue'])
    
    # Načtení vstupního obrazu pro podklad, pokud je k dispozici
    background_vol = None
    input_path = result['input_paths'][0]
    if os.path.exists(input_path):
        try:
            adc_sitk = sitk.ReadImage(input_path)
            background_vol = sitk.GetArrayFromImage(adc_sitk)
            # Normalizace pro zobrazení
            background_vol = (background_vol - background_vol.min()) / (background_vol.max() - background_vol.min() + 1e-8)
        except Exception as e:
            print(f"Nelze načíst pozadí z ADC: {e}")
            background_vol = None
    
    # Vytvoření PDF s více řezy na stránku
    with PdfPages(output_path) as pdf:
        num_pages = math.ceil(num_slices / rows_per_page)
        
        for page in range(num_pages):
            start_idx = page * rows_per_page
            end_idx = min(start_idx + rows_per_page, num_slices)
            slices_on_page = end_idx - start_idx
            
            # Vytvoření mřížky pro aktuální stránku
            fig, axes = plt.subplots(slices_on_page, 2, figsize=(10, 2 * slices_on_page))
            
            # Zajištění, že axes je vždy 2D pole, i když je jen jeden řez
            if slices_on_page == 1:
                axes = np.array([axes])
            
            # Projděme řezy pro aktuální stránku
            for i in range(slices_on_page):
                slice_idx = start_idx + i
                row = i
                
                # Referenční řez (Ground Truth)
                if background_vol is not None:
                    # Zobrazit ADC v pozadí a segmentaci přes něj
                    bg_slice = background_vol[slice_idx, :, :]
                    axes[row, 0].imshow(bg_slice, cmap='gray')
                    axes[row, 0].imshow(reference[slice_idx, :, :], cmap=gt_cmap, alpha=0.7)
                else:
                    # Zobrazit jen segmentaci
                    axes[row, 0].imshow(reference[slice_idx, :, :], cmap='gray')
                
                axes[row, 0].set_title(f'Ground Truth (řez {slice_idx+1})')
                axes[row, 0].axis('off')
                
                # Predikovaný řez (Prediction)
                if background_vol is not None:
                    # Zobrazit ADC v pozadí a segmentaci přes něj
                    bg_slice = background_vol[slice_idx, :, :]
                    axes[row, 1].imshow(bg_slice, cmap='gray')
                    axes[row, 1].imshow(prediction[slice_idx, :, :], cmap=pred_cmap, alpha=0.7)
                else:
                    # Zobrazit jen segmentaci
                    axes[row, 1].imshow(prediction[slice_idx, :, :], cmap='gray')
                
                axes[row, 1].set_title(f'Predikce (řez {slice_idx+1})')
                axes[row, 1].axis('off')
            
            # Přidáme hlavní titulek na stránku
            plt.suptitle(f'Porovnání segmentace - pacient {patient_id}, Dice koeficient: {dice:.3f}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Rezervujeme místo pro hlavní titulek
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"PDF s porovnáním všech řezů uloženo do: {output_path}")
    return output_path


def save_validation_results_pdf(result, output_dir, prefix="validation"):
    """
    Vytvoří PDF soubor se všemi řezy validačního vzorku s třemi samostatnými sloupci: 
    ZADC mapa, LABEL mapa a predikce modelu.
    
    Args:
        result: Výsledek z infer_full_volume nebo infer_full_volume_moe
        output_dir: Výstupní adresář
        prefix: Prefix pro název souboru
        
    Returns:
        str: Cesta k vytvořenému PDF souboru
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction = result['prediction']  # shape (D, H, W)
    reference = result['reference']    # shape (D, H, W)
    
    if reference is None:
        print("Ground truth reference není k dispozici, nelze vytvořit PDF porovnání.")
        return None
    
    # Získání metadat pro název souboru
    metrics = result.get('metrics', {})
    dice = metrics.get('dice', 0.0)
    masd = metrics.get('masd', 0.0)
    nsd = metrics.get('nsd', 0.0)
    patient_id = result.get('patient_id', 'unknown')
    
    # Vytvoření cesty k souboru
    output_name = f"{prefix}_{patient_id}_dice{dice:.3f}.pdf"
    output_path = os.path.join(output_dir, output_name)
    
    # Zjištění počtu řezů a definice barev pro vizualizaci
    num_slices = reference.shape[0]
    rows_per_page = 6  # Počet řezů na stránku
    
    # Definice barevných map pro segmentace
    label_cmap = 'jet'  # Více barev pro label
    pred_cmap = 'jet'   # Více barev pro predikci
    
    # Načtení vstupních obrazů
    zadc_vol = None
    if len(result['input_paths']) > 1:
        try:
            zadc_path = result['input_paths'][1]  # ZADC je druhý vstup
            zadc_sitk = sitk.ReadImage(zadc_path)
            zadc_vol = sitk.GetArrayFromImage(zadc_sitk)
            # Normalizace pro zobrazení
            if np.max(zadc_vol) != np.min(zadc_vol):
                zadc_vol = (zadc_vol - np.min(zadc_vol)) / (np.max(zadc_vol) - np.min(zadc_vol))
        except Exception as e:
            print(f"Nelze načíst ZADC mapu: {e}")
            zadc_vol = None
    
    if zadc_vol is None:
        print("ZADC mapa není k dispozici, používám ADC mapu místo ZADC.")
        try:
            adc_path = result['input_paths'][0]
            adc_sitk = sitk.ReadImage(adc_path)
            zadc_vol = sitk.GetArrayFromImage(adc_sitk)
            # Normalizace pro zobrazení
            if np.max(zadc_vol) != np.min(zadc_vol):
                zadc_vol = (zadc_vol - np.min(zadc_vol)) / (np.max(zadc_vol) - np.min(zadc_vol))
        except Exception as e:
            print(f"Nelze načíst ani ADC mapu: {e}")
            return None
    
    # Vytvoření PDF s více řezy na stránku
    with PdfPages(output_path) as pdf:
        num_pages = math.ceil(num_slices / rows_per_page)
        
        for page in range(num_pages):
            start_idx = page * rows_per_page
            end_idx = min(start_idx + rows_per_page, num_slices)
            slices_on_page = end_idx - start_idx
            
            # Vytvoření mřížky pro aktuální stránku - 3 sloupce: ZADC, LABEL, PRED
            fig, axes = plt.subplots(slices_on_page, 3, figsize=(15, 2 * slices_on_page))
            
            # Zajištění, že axes je vždy 2D pole, i když je jen jeden řez
            if slices_on_page == 1:
                axes = np.array([axes])
            
            # Projděme řezy pro aktuální stránku
            for i in range(slices_on_page):
                slice_idx = start_idx + i
                row = i
                
                # ZADC mapa
                zadc_slice = zadc_vol[slice_idx, :, :]
                axes[row, 0].imshow(zadc_slice, cmap='gray')
                
                # Pro první řez (slice_idx == 0) přidáme informaci o všech metrikách
                if slice_idx == 0:
                    axes[row, 0].set_title(f'ZADC (řez 1) - DICE: {dice:.4f}')
                    axes[row, 1].set_title(f'LABEL (řez 1) - MASD: {masd:.4f}')
                    axes[row, 2].set_title(f'PRED (řez 1) - NSD: {nsd:.4f}')
                else:
                    axes[row, 0].set_title(f'ZADC (řez {slice_idx+1})')
                    axes[row, 1].set_title(f'LABEL (řez {slice_idx+1})')
                    axes[row, 2].set_title(f'PRED (řez {slice_idx+1})')
                
                axes[row, 0].axis('off')
                
                # LABEL mapa (Ground Truth)
                axes[row, 1].imshow(reference[slice_idx, :, :], cmap=label_cmap)
                axes[row, 1].axis('off')
                
                # Predikce modelu
                axes[row, 2].imshow(prediction[slice_idx, :, :], cmap=pred_cmap)
                axes[row, 2].axis('off')
            
            # Přidáme hlavní titulek na stránku
            if page == 0:
                # Na první stránce zobrazíme všechny metriky v hlavním titulku
                plt.suptitle(f'Pacient {patient_id} - Metriky: DICE: {dice:.4f}, MASD: {masd:.4f}, NSD: {nsd:.4f}', 
                             fontsize=14)
            else:
                plt.suptitle(f'Validační vizualizace - pacient {patient_id}', fontsize=14)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Rezervujeme místo pro hlavní titulek
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"PDF s validačními výsledky uloženo do: {output_path}")
    return output_path


def save_segmentation_with_metrics(result, output_dir, prefix="segmentation", save_pdf_comparison=False):
    """
    Uloží segmentaci a přidá hodnoty metrik do názvu souboru.

    Args:
        result: Výsledek z infer_full_volume nebo infer_full_volume_moe
        output_dir: Výstupní adresář
        prefix: Prefix pro název souboru
        save_pdf_comparison: Zda vytvořit PDF s porovnáním řezů
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction = result['prediction']
    input_path = result['input_paths'][0]  # První vstupní cesta pro referenci
    ref_sitk = sitk.ReadImage(input_path)
    
    metrics = result.get('metrics', {})
    dice = metrics.get('dice', 0.0)
    masd = metrics.get('masd', 0.0)
    nsd = metrics.get('nsd', 0.0)
    
    patient_id = result.get('patient_id', 'unknown')
    output_name = f"{prefix}_{patient_id}_dice{dice:.3f}_masd{masd:.3f}_nsd{nsd:.3f}.mha"
    output_path = os.path.join(output_dir, output_name)
    
    save_segmentation_to_file(prediction, ref_sitk, output_path)
    
    # Pokud je požadováno, vytvoříme PDF s porovnáním
    if save_pdf_comparison and result.get('reference') is not None:
        save_slice_comparison_pdf(result, output_dir, prefix=f"{prefix}_comparison")
    
    return output_path


def extract_small_patches(image_data, patch_size, overlap=0.5):
    """
    Extrahuje malé patche z 3D obrazu s daným překryvem.
    
    Args:
        image_data: Tensor se vstupním obrazem, tvar (C, D, H, W)
        patch_size: Tuple s velikostí patche (pz, py, px)
        overlap: Míra překryvu mezi patchi (0-1)
        
    Returns:
        list: Seznam patchů se souřadnicemi [(patch, coords), ...]
    """
    # Získání rozměrů
    c, d, h, w = image_data.shape
    pz, py, px = patch_size
    
    # Výpočet kroku na základě překryvu
    stride_z = int(pz * (1 - overlap))
    stride_y = int(py * (1 - overlap))
    stride_x = int(px * (1 - overlap))
    
    # Zajištění minimálního kroku
    stride_z = max(stride_z, 1)
    stride_y = max(stride_y, 1)
    stride_x = max(stride_x, 1)
    
    # Extrakce patchů
    patches = []
    
    for z in range(0, d-pz+1, stride_z):
        for y in range(0, h-py+1, stride_y):
            for x in range(0, w-px+1, stride_x):
                # Extrakce patche
                patch = image_data[:, z:z+pz, y:y+py, x:x+px]
                coords = (z, y, x, z+pz, y+py, x+px)  # z_start, y_start, x_start, z_end, y_end, x_end
                patches.append((patch, coords))
    
    return patches


def reconstruct_from_patches(patches_with_preds, original_shape, out_channels=2):
    """
    Rekonstruuje obraz z predikcí jednotlivých patchů.
    
    Args:
        patches_with_preds: Seznam patchů s predikcemi a souřadnicemi [(pred, coords), ...]
        original_shape: Tvar původního obrazu (C, D, H, W)
        out_channels: Počet výstupních kanálů v predikci
        
    Returns:
        torch.Tensor: Rekonstruovaný obraz
    """
    _, d, h, w = original_shape
    
    # Inicializace výstupního obrazu a masky váh
    output = torch.zeros((out_channels, d, h, w), dtype=torch.float32)
    weight = torch.zeros((1, d, h, w), dtype=torch.float32)
    
    # Postupné přidávání predikcí patchů
    for pred, coords in patches_with_preds:
        z_start, y_start, x_start, z_end, y_end, x_end = coords
        
        # Přidání predikce do výstupu s váhou
        output[:, z_start:z_end, y_start:y_end, x_start:x_end] += pred
        weight[:, z_start:z_end, y_start:y_end, x_start:x_end] += 1.0
    
    # Normalizace výstupu podle váhy (průměrování překrývajících se částí)
    weight = weight.clamp(min=1.0)  # Zajištění, že všechny hodnoty jsou >= 1
    output = output / weight
    
    return output


def advanced_combine_predictions(main_pred_probs, small_lesion_probs, 
                               alpha=0.6, small_lesion_threshold=0.5, 
                               confidence_boost_factor=1.5, high_conf_threshold=0.8,
                               adaptive_weight=True, size_based_weight=True):
    """
    Pokročilé kombinování predikcí z hlavního modelu a modelu pro malé léze
    s použitím pravděpodobnostních hodnot, adaptivního váhování a lokálního vylepšení.
    
    Args:
        main_pred_probs: Pravděpodobnostní předpovědi hlavního modelu (kanál 1 = léze)
        small_lesion_probs: Pravděpodobnostní předpovědi modelu malých lézí (kanál 1 = léze)
        alpha: Základní váha pro hlavní model (0-1)
        small_lesion_threshold: Prahová hodnota pro detekci malých lézí 
        confidence_boost_factor: Faktor zesílení pro velmi jisté detekce malých lézí
        high_conf_threshold: Práh pro velmi jisté detekce malých lézí (0-1)
        adaptive_weight: Zda použít adaptivní váhování podle velikosti léze
        size_based_weight: Zda použít váhování podle velikosti léze
        
    Returns:
        Kombinovaná pravděpodobnostní mapa a binární předpověď
    """
    import scipy.ndimage as ndimage
    from skimage.measure import label, regionprops
    import torch
    
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(main_pred_probs, torch.Tensor):
        main_pred_probs = main_pred_probs.detach().cpu().numpy()
    
    if isinstance(small_lesion_probs, torch.Tensor):
        small_lesion_probs = small_lesion_probs.detach().cpu().numpy()
    
    # Extrakt relevantních kanálů (pravděpodobnosti léze)
    main_prob = main_pred_probs[1]  # Kanál 1 = léze
    small_prob = small_lesion_probs[1]  # Kanál 1 = léze
    
    # Convert tensor to numpy if needed
    if isinstance(main_prob, torch.Tensor):
        main_prob = main_prob.detach().cpu().numpy()
    
    if isinstance(small_prob, torch.Tensor):
        small_prob = small_prob.detach().cpu().numpy()
    
    # 1. Vytvoření binárních masek pro analýzu komponent
    main_binary = (main_prob > 0.5).astype(np.int32)
    small_binary = (small_prob > small_lesion_threshold).astype(np.int32)
    
    # 2. Analýza velikosti a přesvědčivosti detekovaných objektů
    labeled_small = label(small_binary)
    small_regions = regionprops(labeled_small)
    
    # Vytvoření mapy s vylepšenými váhami pro malé léze
    enhanced_small_weights = np.zeros_like(small_prob)
    confidence_mask = np.zeros_like(small_prob)
    size_weight_mask = np.zeros_like(small_prob)
    
    # Analýza každé detekované malé léze
    for region in small_regions:
        region_mask = labeled_small == region.label
        region_size = region.area
        region_mean_conf = np.mean(small_prob[region_mask])
        
        # Vysoká přesvědčivost = větší váha
        confidence_factor = 1.0
        if region_mean_conf > high_conf_threshold:
            confidence_factor = confidence_boost_factor
            confidence_mask[region_mask] = 1.0
        
        # Malá velikost = větší váha (inverzní vztah k velikosti)
        size_weight = 1.0
        if size_based_weight:
            # Max váha 2.0 pro nejmenší léze (<=10 voxelů), min 0.5 pro léze >100 voxelů
            size_weight = max(0.5, min(2.0, 25.0 / max(10, region_size)))
            size_weight_mask[region_mask] = size_weight
        
        # Aplikace váhy na oblast
        enhanced_small_weights[region_mask] = size_weight * confidence_factor
    
    # 3. Adaptivní kombinace na základě vlastností objektů
    
    # Základní kombinovaná mapa - vážený průměr
    if adaptive_weight:
        # Adaptivní alpha na základě detekovaných vlastností
        adaptive_alpha = np.ones_like(main_prob) * alpha
        
        # Snížení alpha tam, kde jsou malé léze s vysokou přesvědčivostí
        adaptive_alpha[confidence_mask > 0] *= 0.3  # Dáváme větší váhu malým lézím
        
        # Snížení alpha tam, kde jsou velmi malé léze
        adaptive_alpha[size_weight_mask > 1.5] *= 0.4  # Ještě větší váha pro velmi malé léze
        
        # Kombinace s adaptivní alpha
        combined_prob = adaptive_alpha * main_prob + (1 - adaptive_alpha) * small_prob
    else:
        # Standardní kombinace s pevnou alpha
        combined_prob = alpha * main_prob + (1 - alpha) * small_prob
    
    # 4. Rozšíření vlivu vysoce přesvědčivých malých detekcí
    high_conf_small = (small_prob > high_conf_threshold).astype(np.float32)
    if np.any(high_conf_small):
        # Pro velmi přesvědčivé malé léze dáme větší váhu predikci malého modelu
        combined_prob[high_conf_small > 0] = small_prob[high_conf_small > 0]
    
    # 5. Kontextové vylepšení - podpora detekce malých lézí v okolí již detekovaných lézí
    # Aplikujeme pouze tam, kde hlavní model nedetekoval nic
    main_uncertain = (main_prob > 0.25) & (main_prob < 0.5)
    if np.any(main_uncertain):
        # Dilatujeme malé léze pro zachycení okrajových oblastí
        dilated_small = ndimage.binary_dilation(small_binary, iterations=2)
        
        # V nejistých oblastech hlavního modelu, které jsou blízko malých lézí, posílíme predikci
        uncertain_near_small = main_uncertain & dilated_small
        if np.any(uncertain_near_small):
            # Zvýšení pravděpodobnosti v těchto oblastech
            combined_prob[uncertain_near_small] = np.maximum(
                combined_prob[uncertain_near_small],
                small_prob[uncertain_near_small] * 1.2  # Zesílení o 20%
            )
    
    # Vytvoření finální binární předpovědi
    combined_binary = (combined_prob > 0.5).astype(np.int32)
    
    # Pokud by kombinace omezila detekované léze, zachováme detekce z obou modelů (logický OR)
    combined_binary = np.maximum(combined_binary, np.maximum(main_binary, small_binary))
    
    return combined_prob, combined_binary


def infer_full_volume_cascaded(
    input_vol,
    main_model,
    small_lesion_model,
    device="cuda",
    use_tta=False,
    tta_transforms=None,
    cascaded_mode="roi_only",
    small_lesion_threshold=0.5,
    patch_size=(16, 16, 16),
    small_lesion_max_voxels=50,
    alpha=0.6,
    confidence_boost_factor=1.5,
    high_conf_threshold=0.8,
    adaptive_weight=True,
    size_based_weight=True,
    lab_np=None,  # Added lab_np parameter with default value of None
    input_paths=None,  # Přidáme cesty ke vstupním souborům
    label_path=None,    # Přidáme cestu k ground truth souboru
    verbose=False,       # Přidáme parametr verbose pro kontrolu množství výstupů
    training_mode="full_volume",  # Přidáme parametr pro režim tréninku (pro kompatibilitu s validate_one_epoch)
    sw_overlap=0.5  # Přidáme parametr pro překrytí sliding window (pro kompatibilitu s validate_one_epoch)
):
    """
    Provede inferenci na celém objemu s kaskádovým přístupem.
    Nejprve se použije model pro detekci malých lézí, poté se provede inference hlavním modelem.
    Výsledky se kombinují podle zvoleného režimu.

    Args:
        input_vol: Vstupní objem
        main_model: Hlavní model
        small_lesion_model: Model pro detekci malých lézí
        device: Zařízení pro výpočet (cuda nebo cpu)
        use_tta: Použít Test-Time Augmentation
        tta_transforms: Transformace pro TTA
        cascaded_mode: Režim kombinování výsledků
        small_lesion_threshold: Práh pro detekci malých lézí
        patch_size: Velikost patche pro detekci malých lézí
        small_lesion_max_voxels: Maximální počet voxelů pro klasifikaci léze jako 'malé'
        alpha: Váha pro hlavní model při kombinaci výsledků
        confidence_boost_factor: Faktor zvýšení váhy pro detekce s vysokou jistotou
        high_conf_threshold: Práh pravděpodobnosti pro klasifikaci jako 'vysoká jistota'
        adaptive_weight: Použít adaptivní váhování založené na pravděpodobnostech
        size_based_weight: Použít váhování založené na velikosti léze
        lab_np: Ground truth maska pro výpočet metrik (volitelné)
        input_paths: Seznam cest ke vstupním souborům
        label_path: Cesta ke ground truth souboru
        verbose: Kontroluje množství výpisů během inference
        training_mode: Režim tréninku ("full_volume" nebo "patch") pro kompatibilitu s validate_one_epoch
        sw_overlap: Překrytí sliding window pro patch-based inferenci
        
    Returns:
        dict: Slovník s výsledky segmentace a metrikami
    """
    # Začátek diagnostiky
    inference_start = time.time()
    
    if verbose:
        print("\n===== KASKÁDOVÁ INFERENCE =====")
        print(f"Režim: {cascaded_mode}")
        print(f"TTA: {'Ano' if use_tta else 'Ne'}, Max úhel: {len(tta_transforms) if use_tta else 'N/A'}")
        print(f"Model malých lézí - Threshold: {small_lesion_threshold}")
    
    main_model.eval()
    small_lesion_model.eval()
    
    # Příprava input_tensor jednou na začátku
    input_tensor = torch.from_numpy(input_vol).unsqueeze(0).to(device).float()
    
    # 1. Krok: Detekce malých lézí pomocí malého modelu
    if verbose:
        print("\n===== 1. KROK: DETEKCE MALÝCH LÉZÍ =====")
    small_lesion_time_start = time.time()
    
    # Extrakce podvolumů pro detekci malých lézí
    patches_with_coords = extract_small_patches(input_vol, patch_size, overlap=0.5)
    if verbose:
        print(f"Extrakce patchů: {len(patches_with_coords)} patchů o velikosti {patch_size}")
    
    patches_with_preds = []
    
    # Batch processing pro rychlejší inferenci
    batch_size = 32
    for i in range(0, len(patches_with_coords), batch_size):
        batch_patches = []
        batch_coords = []
        
        # Vytvoření dávky
        for j in range(i, min(i + batch_size, len(patches_with_coords))):
            patch, coords = patches_with_coords[j]
            batch_patches.append(patch)
            batch_coords.append(coords)
        
        # Konverze na tensor
        batch_tensor = torch.from_numpy(np.array(batch_patches)).to(device).float()
        
        # Inference
        with torch.no_grad():
            batch_preds = small_lesion_model(batch_tensor)
            batch_preds = torch.sigmoid(batch_preds)
        
        # Uložení predikcí s koordináty
        batch_preds_np = batch_preds.cpu().numpy()
        for pred, coords in zip(batch_preds_np, batch_coords):
            patches_with_preds.append((pred, coords))
    
    # Rekonstrukce pravděpodobnostní mapy z malých patchů
    if verbose:
        print("Rekonstrukce mapy pravděpodobnosti malých lézí...")
    small_lesion_prob_map = reconstruct_from_patches(patches_with_preds, input_vol.shape, out_channels=2)
    
    # Vytvoření binární masky pro malé léze
    if isinstance(small_lesion_prob_map[1], torch.Tensor):
        small_lesion_binary = (small_lesion_prob_map[1].detach().cpu().numpy() > small_lesion_threshold).astype(np.uint8)
    else:
        small_lesion_binary = (small_lesion_prob_map[1] > small_lesion_threshold).astype(np.uint8)
    
    # Časová statistika pro detekci malých lézí
    small_lesion_time = time.time() - small_lesion_time_start
    if verbose:
        print(f"Detekce malých lézí dokončena za {small_lesion_time:.2f}s")
    
    # Základní analýza malých lézí
    small_lesion_voxels = np.sum(small_lesion_binary > 0)
    if verbose:
        print(f"Detekováno {small_lesion_voxels} pozitivních voxelů modelem pro malé léze")
    
    # 2. Krok: Hlavní inference s SwinUNETR modelem
    main_model_time_start = time.time()
    if verbose:
        print("\n===== 2. KROK: HLAVNÍ INFERENCE S SWINUNETR =====")
    
    # 2.1 Nejprve provedeme inferenci samostatného SwinUNETR modelu
    # aby byly výsledky porovnatelné s těmi z validate_one_epoch
    standalone_inference_start = time.time()
    
    # Přizpůsobíme inferenci podle režimu tréninku, přesně jako v validate_one_epoch
    with torch.no_grad():
        if training_mode == "patch":
            try:
                # Import optimalizovaného sliding window
                from monai.inferers import sliding_window_inference
                
                # Použít optimalizovaný sliding window
                logits = sliding_window_inference(
                    inputs=input_tensor, 
                    roi_size=patch_size, 
                    sw_batch_size=1, 
                    predictor=main_model,
                    overlap=sw_overlap,
                    mode="gaussian",  # Pro vážené průměrování překrývajících se oblastí
                    device=device
                )
            except Exception as e:
                print(f"Chyba při sliding window inference: {e}")
                print("Zkouším zpracovat vstup najednou...")
                logits = main_model(input_tensor)
        else:
            # Full volume inference
            logits = main_model(input_tensor)
        
        # Převod logitů na pravděpodobnosti pomocí sigmoid
        probs = torch.sigmoid(logits)
        
        if use_tta and tta_transforms:
            # Test-time augmentation
            preds_list = [probs.cpu()]  # Přidání základní predikce
            
            for transform in tta_transforms:
                # Aplikace transformace
                aug_vol = apply_tta_transform(input_vol, transform)
                transformed_input = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
                
                # Inference
                if training_mode == "patch":
                    try:
                        from monai.inferers import sliding_window_inference
                        aug_logits = sliding_window_inference(
                            inputs=transformed_input, 
                            roi_size=patch_size, 
                            sw_batch_size=1, 
                            predictor=main_model,
                            overlap=sw_overlap,
                            mode="gaussian",
                            device=device
                        )
                    except Exception as e:
                        print(f"Chyba při TTA sliding window inference: {e}")
                        aug_logits = main_model(transformed_input)
                else:
                    aug_logits = main_model(transformed_input)
                
                # Převod logitů na pravděpodobnosti pomocí sigmoid
                aug_probs = torch.sigmoid(aug_logits)
                
                # Inverzní transformace predikce
                inv_probs = invert_tta_transform(aug_probs.cpu().numpy()[0], transform)
                preds_list.append(torch.from_numpy(inv_probs).unsqueeze(0))
            
            # Průměrování predikcí
            mean_probs = torch.mean(torch.cat(preds_list, dim=0), dim=0)
            
            # Převod na binární predikci
            standalone_probs = mean_probs.cpu().numpy()  # [C, D, H, W]
            standalone_pred = (standalone_probs[1] > 0.5).astype(np.uint8)
        else:
            # Standardní inference bez TTA
            standalone_probs = probs.cpu().numpy()[0]  # [C, D, H, W]
            standalone_pred = (standalone_probs[1] > 0.5).astype(np.uint8)
    
    standalone_inference_time = time.time() - standalone_inference_start
    if verbose:
        print(f"Samostatná inference hlavního modelu dokončena za {standalone_inference_time:.2f}s")
    
    # Analýza výsledků hlavního modelu
    standalone_voxels = np.sum(standalone_pred > 0)
    if verbose:
        print(f"Detekováno {standalone_voxels} pozitivních voxelů hlavním modelem")
        
    # 2.2 Nyní provedeme kombinovanou inferenci s kaskádovým přístupem
    if verbose:
        print("\n===== 3. KROK: KOMBINOVANÁ INFERENCE KASKÁDOVÉHO PŘÍSTUPU =====")
    combined_inference_start = time.time()
    
    # Příprava vstupního tensoru pro hlavní model s případným přidáním ROI kanálu
    if cascaded_mode == "roi_only":
        # V tomto režimu přidáme ROI jako další vstupní kanál
        # Příprava ROI z detekce malých lézí (dilatace pro vytvoření kontextu)
        from scipy import ndimage
        roi_mask = ndimage.binary_dilation(small_lesion_binary, iterations=5).astype(np.float32)
        
        # Přidání ROI kanálu ke vstupním datům
        combined_input = np.concatenate([input_vol, roi_mask[np.newaxis, ...]], axis=0)
        combined_input_tensor = torch.from_numpy(combined_input).unsqueeze(0).to(device).float()
        
        if verbose:
            print(f"Přidán ROI kanál ke vstupním datům, nový tvar: {combined_input_tensor.shape}")
    else:
        # V kombinovaném režimu používáme standardní vstupní data
        combined_input_tensor = input_tensor.clone()
    
    # Provádění inference s hlavním modelem stejným způsobem jako u samostatné inference
    with torch.no_grad():
        if training_mode == "patch" and cascaded_mode == "roi_only":
            try:
                from monai.inferers import sliding_window_inference
                logits = sliding_window_inference(
                    inputs=combined_input_tensor, 
                    roi_size=patch_size, 
                    sw_batch_size=1, 
                    predictor=main_model,
                    overlap=sw_overlap,
                    mode="gaussian",
                    device=device
                )
            except Exception as e:
                print(f"Chyba při sliding window inference s ROI: {e}")
                logits = main_model(combined_input_tensor)
        elif training_mode == "patch":
            # Pouze pro kombinovaný režim (stejný vstup jako u samostatné inference)
            # Použijeme již vypočtené probs z předchozího kroku
            pass
        elif cascaded_mode == "roi_only":
            # Full volume inferenci s přidaným ROI kanálem
            logits = main_model(combined_input_tensor)
            probs = torch.sigmoid(logits)
        else:
            # Pro kombinovaný režim použijeme již vypočtené probs z předchozího kroku
            pass
    
    # Predikce s kaskádovým přístupem
    if cascaded_mode == "roi_only":
        # V režimu ROI provedeme novou inferenci s ROI kanálem
        
        # Test-time augmentation pro ROI režim
        if use_tta and tta_transforms:
            preds_list = [probs.cpu()]  # Přidání základní predikce
            
            for transform in tta_transforms:
                # Aplikace transformace na vstupní data včetně ROI kanálu
                aug_combined = apply_tta_transform(combined_input, transform)
                transformed_combined = torch.from_numpy(aug_combined).unsqueeze(0).to(device).float()
                
                # Inference
                if training_mode == "patch":
                    try:
                        from monai.inferers import sliding_window_inference
                        aug_logits = sliding_window_inference(
                            inputs=transformed_combined, 
                            roi_size=patch_size, 
                            sw_batch_size=1, 
                            predictor=main_model,
                            overlap=sw_overlap,
                            mode="gaussian",
                            device=device
                        )
                    except Exception as e:
                        print(f"Chyba při TTA sliding window inference s ROI: {e}")
                        aug_logits = main_model(transformed_combined)
                else:
                    aug_logits = main_model(transformed_combined)
                
                aug_probs = torch.sigmoid(aug_logits)
                
                inv_probs = invert_tta_transform(aug_probs.cpu().numpy()[0], transform)
                preds_list.append(torch.from_numpy(inv_probs).unsqueeze(0))
            
            # Průměrování predikcí
            mean_probs = torch.mean(torch.cat(preds_list, dim=0), dim=0)
            
            # Převod na binární predikci
            cascaded_probs = mean_probs.cpu().numpy()  # [C, D, H, W]
            cascaded_pred = (cascaded_probs[1] > 0.5).astype(np.uint8)
        else:
            # Standardní inference bez TTA
            cascaded_probs = probs.cpu().numpy()[0]  # [C, D, H, W]
            cascaded_pred = (cascaded_probs[1] > 0.5).astype(np.uint8)
        
        # V ROI režimu použijeme přímo výsledek z modelu s ROI kanálem
        final_pred = cascaded_pred
    else:
        # V kombinovaném režimu kombinujeme predikce z obou modelů
        
        # Použití pokročilé funkce pro kombinaci predikcí
        combined_prob, final_pred = advanced_combine_predictions(
            main_pred_probs=standalone_probs,
            small_lesion_probs=small_lesion_prob_map,
            alpha=alpha,
            small_lesion_threshold=small_lesion_threshold,
            confidence_boost_factor=confidence_boost_factor,
            high_conf_threshold=high_conf_threshold,
            adaptive_weight=adaptive_weight,
            size_based_weight=size_based_weight
        )
    
    main_model_time = time.time() - main_model_time_start
    combined_inference_time = time.time() - combined_inference_start
    
    # Omezíme analýzu výsledků na metriky
    if verbose:
        print(f"\n===== SROVNÁNÍ STANDARDNÍ VS. KASKÁDOVÉ PREDIKCE =====")
        standalone_voxels = np.sum(standalone_pred > 0)
        final_voxels = np.sum(final_pred > 0)
        
        print(f"Standardní model: {standalone_voxels} foreground voxelů")
        print(f"Kaskádový model: {final_voxels} foreground voxelů")
        
        # Výpočet přírůstku/úbytku voxelů
        if final_voxels > standalone_voxels:
            added_voxels = final_voxels - standalone_voxels
            print(f"Kaskádou přidáno: {added_voxels} voxelů ({added_voxels/max(1,final_voxels)*100:.1f}% celkové predikce)")
        elif standalone_voxels > final_voxels:
            removed_voxels = standalone_voxels - final_voxels
            print(f"Kaskádou odebráno: {removed_voxels} voxelů ({removed_voxels/max(1,standalone_voxels)*100:.1f}% původní predikce)")
    
    # Omezíme analýzu distribuce velikostí lézí
    if verbose:
        # Analýza distribuce velikostí lézí v obou predikcích
        from skimage.measure import label, regionprops
        labeled_standalone = label(standalone_pred > 0)
        labeled_final = label(final_pred > 0)
        
        regions_standalone = regionprops(labeled_standalone)
        regions_final = regionprops(labeled_final)
        
        print(f"\nPočet lézí ve standardní predikci: {len(regions_standalone)}")
        print(f"Počet lézí v kaskádové predikci: {len(regions_final)}")
        
        if len(regions_standalone) > 0 and len(regions_final) > 0:
            standalone_sizes = [region.area for region in regions_standalone]
            final_sizes = [region.area for region in regions_final]
            
            # Srovnání velikostí
            print(f"\nStandardní predikce - velikosti lézí:")
            print(f"  Min: {min(standalone_sizes)}, Max: {max(standalone_sizes)}, Průměr: {sum(standalone_sizes)/len(standalone_sizes):.1f}")
            
            print(f"Kaskádová predikce - velikosti lézí:")
            print(f"  Min: {min(final_sizes)}, Max: {max(final_sizes)}, Průměr: {sum(final_sizes)/len(final_sizes):.1f}")
            
            # Počet malých lézí v obou predikcích
            standalone_small = len([size for size in standalone_sizes if size <= 50])
            final_small = len([size for size in final_sizes if size <= 50])
            
            print(f"\nMalé léze (<=50 voxelů):")
            print(f"  Standardní predikce: {standalone_small} lézí ({standalone_small/len(regions_standalone)*100:.1f}%)")
            print(f"  Kaskádová predikce: {final_small} lézí ({final_small/len(regions_final)*100:.1f}%)")
            print(f"  Rozdíl: {final_small - standalone_small} lézí")
    
    # Výpočet metrik, pokud je k dispozici ground truth
    metrics = {}
    standalone_metrics = {}
    
    if lab_np is not None:
        from src.utils.metrics import compute_all_metrics
        
        # Vždy vypíšeme srovnání metrik, bez ohledu na verbose parametr
        print(f"\n===== SROVNÁNÍ METRIK STANDARDNÍ VS. KASKÁDOVÉ METODY =====")
        
        # Metriky pro standardní model
        standalone_metrics = compute_all_metrics(standalone_pred, lab_np, include_surface_metrics=True)
        print(f"Standardní model (pouze SwinUNETR):")
        print(f"  DICE: {standalone_metrics.get('dice', 0):.4f}")
        print(f"  MASD: {standalone_metrics.get('masd', 0):.4f}")
        print(f"  NSD: {standalone_metrics.get('nsd', 0):.4f}")
        
        # Metriky pro kaskádový model
        metrics = compute_all_metrics(final_pred, lab_np, include_surface_metrics=True)
        print(f"\nKaskádový model (kombinovaný přístup):")
        print(f"  DICE: {metrics.get('dice', 0):.4f}")
        print(f"  MASD: {metrics.get('masd', 0):.4f}")
        print(f"  NSD: {metrics.get('nsd', 0):.4f}")
        
        # Srovnání hlavních metrik
        dice_diff = metrics.get("dice", 0) - standalone_metrics.get("dice", 0)
        masd_diff = standalone_metrics.get("masd", 0) - metrics.get("masd", 0)  # MASD nižší je lepší
        nsd_diff = metrics.get("nsd", 0) - standalone_metrics.get("nsd", 0)
        
        print(f"\nRozdíl v metrikách (kaskádový - standardní):")
        print(f"  DICE: {dice_diff:.4f} {'(zlepšení)' if dice_diff > 0 else '(zhoršení)' if dice_diff < 0 else '(beze změny)'}")
        print(f"  MASD: {-masd_diff:.4f} {'(zlepšení)' if masd_diff > 0 else '(zhoršení)' if masd_diff < 0 else '(beze změny)'}")
        print(f"  NSD: {nsd_diff:.4f} {'(zlepšení)' if nsd_diff > 0 else '(zhoršení)' if nsd_diff < 0 else '(beze změny)'}")

    # Získání ID pacienta, pokud je k dispozici label_path
    patient_id = None
    if label_path:
        try:
            from src.data.dataset import get_subject_id_from_filename
            patient_id = get_subject_id_from_filename(label_path)
        except:
            # Fallback pokud není k dispozici funkce get_subject_id_from_filename
            patient_id = os.path.basename(label_path).split('_')[0]

    # Vytvoření výsledného slovníku
    result = {
        "prediction": final_pred,
        "standalone_prediction": standalone_pred,  # Přidána samostatná predikce hlavního modelu
        "reference": lab_np,  # Přidání reference
        "input_paths": input_paths if input_paths else [],  # Přidání cest ke vstupním souborům
        "label_path": label_path,  # Přidání cesty k ground truth souboru
        "metrics": metrics,
        "standalone_metrics": standalone_metrics,  # Přidány metriky samostatného modelu
        "small_lesion_metrics": {},
        "timing": {
            "small_lesion_model": small_lesion_time,
            "main_model": main_model_time,
            "standalone_inference": standalone_inference_time,
            "combined_inference": combined_inference_time,
            "total": small_lesion_time + main_model_time
        }
    }
    
    # Přidání patient_id, pokud bylo získáno
    if patient_id:
        result['patient_id'] = patient_id
    
    # Přidání diagnositických informací
    result["diagnostics"] = {
        "processed_shape": input_vol.shape,
        "foreground_voxels": np.sum(final_pred > 0),
        "standalone_foreground_voxels": np.sum(standalone_pred > 0),
        "small_lesion_foreground_voxels": np.sum(small_lesion_binary > small_lesion_threshold),
        "cascaded_mode": cascaded_mode
    }
    
    return result