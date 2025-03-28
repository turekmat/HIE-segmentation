import os
import torch
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import time

from ..utils.metrics import dice_coefficient, compute_masd, compute_nsd
from ..data.preprocessing import get_tta_transforms, apply_tta_transform, invert_tta_transform
from ..data.dataset import extract_patient_id

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
    
    # Extrakt relevantních kanálů (pravděpodobnosti léze)
    main_prob = main_pred_probs[1]  # Kanál 1 = léze
    small_prob = small_lesion_probs[1]  # Kanál 1 = léze
    
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
    size_based_weight=True
):
    """
    Provádí kaskádovanou inferenci na celém objemu s využitím hlavního modelu a modelu pro malé léze.
    
    Args:
        input_vol (numpy.ndarray): Vstupní objem tvaru (C, D, H, W)
        main_model (torch.nn.Module): Hlavní model pro segmentaci
        small_lesion_model (torch.nn.Module): Model specializovaný na malé léze
        device (str): Zařízení, na kterém běží modely ('cuda' nebo 'cpu')
        use_tta (bool): Zda použít test-time augmentaci
        tta_transforms (list): Seznam transformací pro TTA
        cascaded_mode (str): Režim kaskády - 'roi_only' nebo 'combined' 
                            - 'roi_only': Použije model malých lézí jen pro oblast zájmu kolem hlavní predikce
                            - 'combined': Spojí predikce hlavního modelu a modelu malých lézí
        small_lesion_threshold (float): Práh pravděpodobnosti pro detekci malých lézí
        patch_size (tuple): Velikost patche pro model malých lézí
        small_lesion_max_voxels (int): Maximální počet voxelů pro klasifikaci jako malé léze
        alpha (float): Základní váha pro hlavní model při kombinaci predikcí (0.0-1.0)
        confidence_boost_factor (float): Faktor zvýšení váhy pro velmi jisté detekce
        high_conf_threshold (float): Práh pro klasifikaci jako velmi jistá detekce
        adaptive_weight (bool): Zda použít adaptivní váhování podle velikosti a jistoty detekce
        size_based_weight (bool): Zda použít váhování podle velikosti léze
        
    Returns:
        numpy.ndarray: Finální predikce jako binární maska
        dict: Metriky a časy inferenčního procesu
    """
    # Začátek diagnostiky
    inference_start = time.time()
    
    print("\n===== KASKÁDOVÁ INFERENCE =====")
    print(f"Režim: {cascaded_mode}")
    print(f"Vstupní soubory: {input_vol}")
    print(f"TTA: {'Ano' if use_tta else 'Ne'}, Max úhel: {tta_transforms if use_tta else 'N/A'}")
    print(f"Model malých lézí - Threshold: {small_lesion_threshold}")
    
    main_model.eval()
    small_lesion_model.eval()
    
    # 1. Krok: Detekce malých lézí pomocí malého modelu
    print("\n===== 1. KROK: DETEKCE MALÝCH LÉZÍ =====")
    small_lesion_time_start = time.time()
    
    # Extrakce podvolumů pro detekci malých lézí
    patches_with_coords = extract_small_patches(input_vol, patch_size, overlap=0.5)
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
    
    # Rekonstrukce pravděpodobnostní mapy
    print("Rekonstrukce mapy pravděpodobnosti malých lézí...")
    small_lesion_prob_map = np.zeros((2, *input_vol.shape[1:]))  # Two channels for binary segmentation
    small_lesion_count = np.zeros((1, *input_vol.shape[1:]))     # Count for averaging
    
    for pred, (z_start, z_end, y_start, y_end, x_start, x_end) in patches_with_preds:
        small_lesion_prob_map[:, z_start:z_end, y_start:y_end, x_start:x_end] += pred
        small_lesion_count[0, z_start:z_end, y_start:y_end, x_start:x_end] += 1
    
    # Průměrování překrývajících se oblastí
    small_lesion_count = np.maximum(small_lesion_count, 1)  # Vyhnutí se dělení nulou
    small_lesion_prob_map /= small_lesion_count
    
    # Vytvoření binární masky pro detekci malých lézí
    small_lesion_binary = (small_lesion_prob_map[1] > small_lesion_threshold).astype(np.int32)
    
    # Konverze na tensor pro další zpracování
    small_lesion_mask = torch.from_numpy(small_lesion_prob_map).unsqueeze(0).to(device).float()
    
    small_lesion_time = time.time() - small_lesion_time_start
    
    # Analýza výsledků modelu malých lézí
    small_lesion_voxels = np.sum(small_lesion_binary > 0)
    print(f"Detekce malých lézí dokončena za {small_lesion_time:.2f}s")
    print(f"Detekováno {small_lesion_voxels} voxelů malých lézí")
    
    # Analýza detekovaných malých lézí
    if small_lesion_voxels > 0:
        from skimage.measure import label, regionprops
        labeled_small = label(small_lesion_binary > 0)
        regions_small = regionprops(labeled_small)
        
        print(f"Počet spojitých malých lézí: {len(regions_small)}")
        
        lesion_sizes = [region.area for region in regions_small]
        if lesion_sizes:
            print(f"Minimální velikost detekované léze: {min(lesion_sizes)} voxelů")
            print(f"Maximální velikost detekované léze: {max(lesion_sizes)} voxelů")
            print(f"Průměrná velikost detekované léze: {sum(lesion_sizes)/len(lesion_sizes):.1f} voxelů")
            
            # Distribuce velikostí detekovaných lézí
            size_buckets = {
                "1-5 voxelů": 0,
                "6-20 voxelů": 0,
                "21-50 voxelů": 0,
                ">50 voxelů": 0
            }
            
            for size in lesion_sizes:
                if 1 <= size <= 5:
                    size_buckets["1-5 voxelů"] += 1
                elif 6 <= size <= 20:
                    size_buckets["6-20 voxelů"] += 1
                elif 21 <= size <= 50:
                    size_buckets["21-50 voxelů"] += 1
                else:
                    size_buckets[">50 voxelů"] += 1
            
            print("\nDistribuce velikostí detekovaných malých lézí:")
            for category, count in size_buckets.items():
                percentage = (count / len(regions_small)) * 100
                print(f"  {category}: {count} lézí ({percentage:.1f}%)")
    
        # Pokud máme ground truth, spočítáme přesnost detekce malých lézí
        if lab_np is not None:
            # Překryv s ground truth
            intersection = np.sum((small_lesion_binary > 0) & (lab_np > 0))
            union = np.sum((small_lesion_binary > 0) | (lab_np > 0))
            dice_small = (2.0 * intersection) / (np.sum(small_lesion_binary > 0) + np.sum(lab_np > 0)) if union > 0 else 0
            
            print(f"\nPřesnost modelu malých lézí:")
            print(f"  DICE koeficient: {dice_small:.4f}")
            print(f"  True Positive voxelů: {intersection} voxelů")
            print(f"  False Positive voxelů: {np.sum((small_lesion_binary > 0) & (lab_np == 0))} voxelů")
            print(f"  False Negative voxelů: {np.sum((small_lesion_binary == 0) & (lab_np > 0))} voxelů")
    
    # 2. Krok: Finální segmentace s hlavním modelem
    print("\n===== 2. KROK: FINÁLNÍ SEGMENTACE S HLAVNÍM MODELEM =====")
    main_model_time_start = time.time()
    
    # Nejprve spočítáme standardní inferenci s hlavním modelem pro srovnání
    standard_pred = None
    with torch.no_grad():
        if use_tta:
            # TTA inference hlavního modelu
            avg_probs_main = 0
            for transform in tta_transforms:
                aug_vol = apply_tta_transform(input_vol, transform)
                aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
                
                pred_logits = main_model(aug_tensor)
                
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(pred_logits)
                probs_np = probs.cpu().numpy()[0]
                inv_probs = invert_tta_transform(probs_np, transform)
                avg_probs_main += inv_probs
            
            avg_probs_main /= len(tta_transforms)
            standard_pred = np.argmax(avg_probs_main, axis=0)
        else:
            # Standardní inference
            pred_logits = main_model(input_tensor)
            standard_pred = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
    
    # Kombinace vstupního obrazu s mapou malých lézí (přidání jako další kanál)
    if cascaded_mode == "roi_only":
        # Použijeme pouze ROI masku jako další vstupní kanál
        augmented_input = torch.cat([
            torch.from_numpy(input_vol), 
            small_lesion_mask
        ], dim=0).unsqueeze(0).to(device)
        
        print(f"ROI režim: Přidán kanál s malými lézemi k vstupnímu tensoru, nový tvar: {augmented_input.shape}")
        
        # Inference s augmentovaným vstupem
        with torch.no_grad():
            if use_tta:
                # TTA inference
                avg_probs = 0
                for transform in tta_transforms:
                    aug_vol = apply_tta_transform(augmented_input.cpu().numpy()[0], transform)
                    aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
                    
                    pred_logits = main_model(aug_tensor)
                    
                    softmax = torch.nn.Softmax(dim=1)
                    probs = softmax(pred_logits)
                    probs_np = probs.cpu().numpy()[0]
                    inv_probs = invert_tta_transform(probs_np, transform)
                    avg_probs += inv_probs
                
                avg_probs /= len(tta_transforms)
                final_pred = np.argmax(avg_probs, axis=0)
            else:
                # Standardní inference
                pred_logits = main_model(augmented_input)
                final_pred = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
    
    else:  # "combined" mode
        # Pokročilá kombinace predikcí hlavního modelu a modelu malých lézí
        print(f"Kombinovaný režim: Pokročilé spojení predikcí hlavního modelu a modelu malých lézí")
        
        # Získání pravděpodobnostních map místo binárních predikcí
        # Pro hlavní model
        if use_tta:
            # TTA inference hlavního modelu pro získání pravděpodobnostních map
            avg_probs_main = np.zeros((2, *input_vol.shape[1:]))  # Background a foreground kanály
            for transform in tta_transforms:
                aug_vol = apply_tta_transform(input_vol, transform)
                aug_tensor = torch.from_numpy(aug_vol).unsqueeze(0).to(device).float()
                
                pred_logits = main_model(aug_tensor)
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(pred_logits)
                probs_np = probs.cpu().numpy()[0]
                inv_probs = invert_tta_transform(probs_np, transform)
                avg_probs_main += inv_probs
            
            avg_probs_main /= len(tta_transforms)
            main_pred_probs = avg_probs_main
        else:
            # Standardní inference
            pred_logits = main_model(input_tensor)
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(pred_logits)
            main_pred_probs = probs.cpu().numpy()[0]
        
        # Pro model malých lézí - převedení na pravděpodobnostní mapu
        # Již máme small_lesion_prob_map z předchozího kroku inference
        
        # Použití pokročilé funkce pro kombinaci
        print(f"Používám pokročilou funkci pro kombinaci predikcí...")
        combined_prob, combined_pred = advanced_combine_predictions(
            main_pred_probs=main_pred_probs,
            small_lesion_probs=small_lesion_prob_map,
            alpha=alpha,  # Parametr přes argumenty funkce
            small_lesion_threshold=small_lesion_threshold,
            confidence_boost_factor=confidence_boost_factor,  # Parametr přes argumenty funkce
            high_conf_threshold=high_conf_threshold,          # Parametr přes argumenty funkce
            adaptive_weight=adaptive_weight,                  # Parametr přes argumenty funkce
            size_based_weight=size_based_weight              # Parametr přes argumenty funkce
        )
        
        # Použití kombinované predikce
        final_pred = combined_pred
        
        # Analýza vlivu pokročilé kombinace
        standard_pred_binary = np.argmax(main_pred_probs, axis=0)
        small_pred_binary = (small_lesion_prob_map[1] > small_lesion_threshold).astype(np.int32)
        
        # Výpočet statistik o příspěvku každému modelu
        unique_main = np.sum((standard_pred_binary > 0) & (final_pred > 0))
        unique_small = np.sum((small_pred_binary > 0) & (final_pred > 0))
        total_voxels = np.sum(final_pred > 0)
        
        if total_voxels > 0:
            print(f"Statistika příspěvku modelů v pokročilé kombinaci:")
            print(f"  Hlavní model: {unique_main} voxelů ({unique_main/total_voxels*100:.1f}%)")
            print(f"  Model malých lézí: {unique_small} voxelů ({unique_small/total_voxels*100:.1f}%)")
            
            # Detekce, kde pokročilá kombinace přidala něco navíc
            advanced_added = np.sum((final_pred > 0) & (standard_pred_binary == 0) & (small_pred_binary == 0))
            if advanced_added > 0:
                print(f"  Pokročilá kombinace přidala: {advanced_added} voxelů ({advanced_added/total_voxels*100:.1f}%)")
        
        # Uložíme si original binary predikce pro pozdější analýzu
        standard_pred = standard_pred_binary
    
    main_model_time = time.time() - main_model_time_start
    print(f"Inference hlavního modelu dokončena za {main_model_time:.2f}s")
    
    # Analýza výsledků hlavního modelu a kombinované predikce
    if standard_pred is not None:
        standard_voxels = np.sum(standard_pred > 0)
        final_voxels = np.sum(final_pred > 0)
        
        print(f"\n===== SROVNÁNÍ STANDARDNÍ VS. KASKÁDOVÉ PREDIKCE =====")
        print(f"Standardní model: {standard_voxels} foreground voxelů")
        print(f"Kaskádový model: {final_voxels} foreground voxelů")
        
        if final_voxels > standard_voxels:
            added_voxels = final_voxels - standard_voxels
            print(f"Kaskádou přidáno: {added_voxels} voxelů ({added_voxels/max(1,final_voxels)*100:.1f}% celkové predikce)")
        
        # Analýza distribuce velikostí lézí v obou predikcích
        from skimage.measure import label, regionprops
        labeled_standard = label(standard_pred > 0)
        labeled_final = label(final_pred > 0)
        
        regions_standard = regionprops(labeled_standard)
        regions_final = regionprops(labeled_final)
        
        print(f"\nPočet lézí ve standardní predikci: {len(regions_standard)}")
        print(f"Počet lézí v kaskádové predikci: {len(regions_final)}")
        
        if len(regions_standard) > 0 and len(regions_final) > 0:
            standard_sizes = [region.area for region in regions_standard]
            final_sizes = [region.area for region in regions_final]
            
            # Srovnání velikostí
            print(f"\nStandardní predikce - velikosti lézí:")
            print(f"  Min: {min(standard_sizes)}, Max: {max(standard_sizes)}, Průměr: {sum(standard_sizes)/len(standard_sizes):.1f}")
            
            print(f"Kaskádová predikce - velikosti lézí:")
            print(f"  Min: {min(final_sizes)}, Max: {max(final_sizes)}, Průměr: {sum(final_sizes)/len(final_sizes):.1f}")
            
            # Počet malých lézí v obou predikcích
            standard_small = len([size for size in standard_sizes if size <= 50])
            final_small = len([size for size in final_sizes if size <= 50])
            
            print(f"\nMalé léze (<=50 voxelů):")
            print(f"  Standardní predikce: {standard_small} lézí ({standard_small/len(regions_standard)*100:.1f}%)")
            print(f"  Kaskádová predikce: {final_small} lézí ({final_small/len(regions_final)*100:.1f}%)")
            print(f"  Rozdíl: {final_small - standard_small} lézí")
    
    # Výpočet metrik, pokud je k dispozici ground truth
    metrics = {}
    if lab_np is not None:
        from src.utils.metrics import compute_all_metrics
        metrics = compute_all_metrics(final_pred, lab_np, include_surface_metrics=True)
        print(f"\n===== METRIKY =====")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    # Vytvoření výsledného slovníku
    result = {
        "prediction": final_pred,
        "metrics": metrics,
        "small_lesion_metrics": {},  # TODO: Přidat metriky pro model malých lézí
        "timing": {
            "small_lesion_model": small_lesion_time,
            "main_model": main_model_time,
            "total": small_lesion_time + main_model_time
        }
    }
    
    # Přidání diagnositických informací
    result["diagnostics"] = {
        "processed_shape": input_vol.shape,
        "foreground_voxels": np.sum(final_pred > 0),
        "small_lesion_foreground_voxels": np.sum(small_lesion_binary > small_lesion_threshold),
        "cascaded_mode": cascaded_mode
    }
    
    return result 