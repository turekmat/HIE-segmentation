import os
import torch
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

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