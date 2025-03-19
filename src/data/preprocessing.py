import os
import numpy as np
import SimpleITK as sitk
import random
import re
from scipy.ndimage import rotate, gaussian_filter, distance_transform_edt, binary_erosion
import torch

def filter_augmented_files(file_list, max_aug):
    """
    Filtruje augmentované soubory podle původního souboru.
    Zachovává originální soubory a omezuje počet augmentovaných
    souborů na max_aug na jeden originální soubor.

    Args:
        file_list: Seznam souborů k filtrování
        max_aug: Maximální počet augmentovaných souborů na jeden originální

    Returns:
        Seznam filtrovaných souborů
    """
    grouped = {}
    for f in file_list:
        key = get_base_id(f)
        if '_aug' in f.lower():
            # Soubor je augmentovaný
            if key not in grouped:
                grouped[key] = {'orig': None, 'aug': []}
            grouped[key]['aug'].append(f)
        else:
            # Soubor bez "_aug" je považován za originální
            if key not in grouped:
                grouped[key] = {'orig': None, 'aug': []}
            grouped[key]['orig'] = f

    filtered_list = []
    for key in sorted(grouped.keys()):
        entry = grouped[key]
        if entry['orig'] is not None:
            filtered_list.append(entry['orig'])
            selected_aug = sorted(entry['aug'])[:max_aug]
            filtered_list.extend(selected_aug)
        else:
            selected = sorted(entry['aug'])[:max_aug]
            filtered_list.extend(selected)
    return filtered_list


def get_base_id(filename: str):
    """
    Získá základní ID souboru bez sufixu _aug.
    """
    filename_lower = filename.lower()
    if '_aug' in filename_lower:
        return re.sub(r'_aug\d+.*', '', filename_lower)
    else:
        return filename_lower


def random_3d_augmentation(
    adc_np,
    zadc_np,
    label_np,
    angle_max=3,
    p_flip=0.5,
    p_noise=0.0,
    noise_std=0.01,
    p_smooth=0.0,
    smooth_sigma=1.0
):
    """
    Provádí augmentaci 3D objemů.

    Args:
        adc_np: Numpy array ADC objemu
        zadc_np: Numpy array Z-ADC objemu
        label_np: Numpy array masky
        angle_max: Maximální úhel rotace
        p_flip: Pravděpodobnost horizontálního flipu
        p_noise: Pravděpodobnost přidání šumu
        noise_std: Směrodatná odchylka pro šum
        p_smooth: Pravděpodobnost vyhlazení
        smooth_sigma: Sigma pro Gaussovské vyhlazení

    Returns:
        Augmentované objemy (adc_np, zadc_np, label_np)
    """
    # Horizontální flip
    if random.random() < p_flip:
      adc_np = np.flip(adc_np, axis=2).copy()
      zadc_np = np.flip(zadc_np, axis=2).copy()
      label_np = np.flip(label_np, axis=2).copy()

    # Náhodná rotace
    angle = random.uniform(-angle_max, angle_max)
    axes  = random.choice([(0, 1), (0, 2), (1, 2)])  # vybereme jednu dvojici os
    adc_np = rotate(adc_np,   angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    zadc_np = rotate(zadc_np, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    label_np = rotate(label_np, angle=angle, axes=axes, reshape=False, order=0, mode='nearest')

    # Gaussovský šum
    if random.random() < p_noise:
        noise_adc = np.random.normal(0, noise_std, size=adc_np.shape)
        noise_z   = np.random.normal(0, noise_std, size=zadc_np.shape)
        adc_np   = adc_np + noise_adc
        zadc_np  = zadc_np + noise_z

    # Gaussovské vyhlazení
    if random.random() < p_smooth:
        adc_np   = gaussian_filter(adc_np, sigma=smooth_sigma)
        zadc_np  = gaussian_filter(zadc_np, sigma=smooth_sigma)

    return adc_np, zadc_np, label_np


def prepare_preprocessed_data(
    adc_folder, z_folder, label_folder,
    output_adc, output_z, output_label,
    normalize=True,
    allow_normalize_spacing=False
):
    """
    Předzpracovává data z původních složek do výstupních složek.
    Implementace založená na optimalizovaném předzpracování.
    
    Args:
        adc_folder: Vstupní složka s ADC skeny
        z_folder: Vstupní složka s Z-ADC skeny
        label_folder: Vstupní složka s maskami
        output_adc: Výstupní složka pro předzpracované ADC skeny
        output_z: Výstupní složka pro předzpracované Z-ADC skeny
        output_label: Výstupní složka pro předzpracované masky
        normalize: Zda normalizovat intenzity (True/False)
        allow_normalize_spacing: Zda normalizovat prostorové kroky (True/False)
    """
    # Kontrola a vytvoření výstupních adresářů
    for output_dir in [output_adc, output_z, output_label]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Kontrola, zda již existují předzpracované soubory
    existing_files = [
        len([f for f in os.listdir(dir) if f.endswith('.mha')]) 
        for dir in [output_adc, output_z, output_label]
    ]
    
    if all(count > 0 for count in existing_files) and len(set(existing_files)) == 1:
        print("Preprocessed data already exist. Skipping preprocessing.")
        return
    
    # Načtení seznamu souborů
    adc_files = sorted([f for f in os.listdir(adc_folder) if f.endswith('.mha')])
    z_files = sorted([f for f in os.listdir(z_folder) if f.endswith('.mha')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mha')])
    
    if not (len(adc_files) == len(z_files) == len(label_files)):
        raise ValueError("Inconsistent number of input files across folders.")
    
    print(f"Processing {len(adc_files)} file sets...")
    
    # Zpracování každého souboru
    for i, (adc_name, zadc_name, label_name) in enumerate(zip(adc_files, z_files, label_files)):
        print(f"Processing {i+1}/{len(adc_files)}: {adc_name}")
        
        # Načtení vstupních skenů
        adc_img = sitk.ReadImage(os.path.join(adc_folder, adc_name))
        zadc_img = sitk.ReadImage(os.path.join(z_folder, zadc_name))
        label_img = sitk.ReadImage(os.path.join(label_folder, label_name))
        
        # 1. Normalizace spacing - jako první krok pokud je povoleno
        if allow_normalize_spacing:
            adc_img = normalize_spacing(adc_img)
            zadc_img = normalize_spacing(zadc_img)
            label_img = normalize_spacing(label_img)
        
        # Uložení metadat pro pozdější použití
        final_spacing = adc_img.GetSpacing()
        final_direction = adc_img.GetDirection()
        final_origin = adc_img.GetOrigin()
        
        # 2. Převod na numpy arrays
        adc_np = sitk.GetArrayFromImage(adc_img)
        zadc_np = sitk.GetArrayFromImage(zadc_img)
        label_np = sitk.GetArrayFromImage(label_img)
        
        # 3. Vytvoření masky mozku a normalizace
        if normalize:
            brain_mask = (adc_np > 0) & (zadc_np > 0)
            adc_np = z_score_normalize(adc_np, brain_mask)
            zadc_np = z_score_normalize(zadc_np, brain_mask)
        
        # 4. Výpočet a aplikace bounding boxu
        bounding_box = compute_largest_3d_bounding_box([adc_np, zadc_np], threshold=0)
        adc_np, zadc_np, label_np = crop_to_largest_bounding_box(adc_np, zadc_np, label_np, bounding_box, margin=5)
        
        # 5. Padding na násobky 32
        adc_np = pad_3d_all_dims_to_multiple_of_32(adc_np)
        zadc_np = pad_3d_all_dims_to_multiple_of_32(zadc_np)
        label_np = pad_3d_all_dims_to_multiple_of_32(label_np)
        
        # Převod zpět na SimpleITK objemy
        processed_adc = sitk.GetImageFromArray(adc_np)
        processed_z = sitk.GetImageFromArray(zadc_np)
        processed_label = sitk.GetImageFromArray(label_np)
        
        # Nastavení metadat
        for img in [processed_adc, processed_z, processed_label]:
            img.SetSpacing(final_spacing)
            img.SetDirection(final_direction)
            img.SetOrigin(final_origin)
        
        # Uložení předzpracovaných souborů
        sitk.WriteImage(processed_adc, os.path.join(output_adc, adc_name))
        sitk.WriteImage(processed_z, os.path.join(output_z, zadc_name))
        sitk.WriteImage(processed_label, os.path.join(output_label, label_name))
        
        # Uvolnění paměti
        del adc_np, zadc_np, label_np, processed_adc, processed_z, processed_label
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Preprocessing complete.")


def compute_largest_3d_bounding_box(volumes, threshold=0):
    """
    Vypočítá největší bounding box ze všech vstupních objemů.
    """
    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])

    for volume in volumes:
        nonzero = np.argwhere(volume > threshold)
        if nonzero.size > 0:
            min_coords = np.minimum(min_coords, nonzero.min(axis=0))
            max_coords = np.maximum(max_coords, nonzero.max(axis=0) + 1)

    return ((int(min_coords[0]), int(max_coords[0])),
            (int(min_coords[1]), int(max_coords[1])),
            (int(min_coords[2]), int(max_coords[2])))


def crop_to_largest_bounding_box(adc_np, zadc_np, label_np, bounding_box, margin=5):
    """
    Ořízne objemy podle bounding boxu s přidaným okrajem.
    """
    (minD, maxD), (minH, maxH), (minW, maxW) = bounding_box

    # Přidání okraje
    minD = max(0, minD - margin)
    maxD = min(adc_np.shape[0], maxD + margin)
    minH = max(0, minH - margin)
    maxH = min(adc_np.shape[1], maxH + margin)
    minW = max(0, minW - margin)
    maxW = min(adc_np.shape[2], maxW + margin)

    # Oříznutí objemů
    return (adc_np[minD:maxD, minH:maxH, minW:maxW],
            zadc_np[minD:maxD, minH:maxH, minW:maxW],
            label_np[minD:maxD, minH:maxH, minW:maxW])


def pad_3d_all_dims_to_multiple_of_32(volume_3d, mode="edge"):
    """
    Přidá padding k 3D objemu tak, aby všechny rozměry byly násobky 32.
    """
    def pad_dim_to_32(dim_size):
        return ((dim_size - 1) // 32 + 1) * 32 if dim_size % 32 != 0 else dim_size

    newD, newH, newW = [pad_dim_to_32(dim) for dim in volume_3d.shape]
    padD, padH, padW = newD - volume_3d.shape[0], newH - volume_3d.shape[1], newW - volume_3d.shape[2]

    return np.pad(
        volume_3d,
        pad_width=((0, padD), (0, padH), (0, padW)),
        mode=mode
    )


def normalize_spacing(image_sitk, target_spacing=(1.0, 1.0, 1.0)):
    """
    Normalizuje spacing obrazu na cílový spacing.
    """
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    new_size = [int(round(osz * ospacing / tspacing))
                for osz, ospacing, tspacing in zip(original_size, original_spacing, target_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    # Interpolace: lineární pro obrazy, nearest neighbor pro masky
    if image_sitk.GetPixelID() in [sitk.sitkUInt8, sitk.sitkInt8]:  # pravděpodobně maska
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:  # pravděpodobně obraz
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image_sitk)


def z_score_normalize(image_np, mask=None):
    """
    Provádí Z-score normalizaci objemu (odečtení průměru a vydělení směrodatnou odchylkou).
    Pokud je poskytnutá maska, normalizace se provádí pouze na základě voxelů v masce.
    """
    valid_voxels = image_np[mask > 0] if mask is not None else image_np
    mean, std = np.mean(valid_voxels), np.std(valid_voxels)
    return (image_np - mean) / (std if std != 0 else 1.0)


def apply_tta_transform(volume, transform):
    """
    Aplikuje transformaci pro Test-Time Augmentaci.
    """
    do_flip = transform.get('flip', False)
    rotation = transform.get('rotation', None)
    
    # Pokud máme (C, D, H, W), pracujeme s každým kanálem zvlášť
    if len(volume.shape) == 4:
        C, D, H, W = volume.shape
        transformed = np.zeros_like(volume)
        for c in range(C):
            transformed[c] = apply_tta_transform(volume[c], transform)
        return transformed
    
    # Pro samostatný objem (D, H, W)
    result = volume.copy()
    
    # Horizontální flip (pokud je požadován)
    if do_flip:
        result = np.flip(result, axis=2)
    
    # Rotace (pokud je požadována)
    if rotation is not None:
        angle = rotation['angle']
        axes = rotation['axes']
        result = rotate(result, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    
    return result


def invert_tta_transform(volume, transform):
    """
    Invertuje transformaci TTA, aby se výsledky mohly průměrovat.
    """
    do_flip = transform.get('flip', False)
    rotation = transform.get('rotation', None)
    
    # Pokud máme (C, D, H, W), pracujeme s každým kanálem zvlášť
    if len(volume.shape) == 4:
        C, D, H, W = volume.shape
        inverted = np.zeros_like(volume)
        for c in range(C):
            inverted[c] = invert_tta_transform(volume[c], transform)
        return inverted
    
    result = volume.copy()
    
    # Invertujeme rotaci (pokud byla aplikována)
    if rotation is not None:
        angle = -rotation['angle']  # Obrátit úhel
        axes = rotation['axes']
        result = rotate(result, angle=angle, axes=axes, reshape=False, order=1, mode='nearest')
    
    # Invertujeme flip (pokud byl aplikován)
    if do_flip:
        result = np.flip(result, axis=2)
    
    return result


def get_tta_transforms(angle_max=3):
    """
    Vytvoří seznam transformací pro Test-Time Augmentaci.
    """
    tta_transforms = []
    flips = [False, True]
    rotations = [None,
                 {"angle": angle_max,  "axes": (0, 1)},
                 {"angle": -angle_max, "axes": (0, 1)},
                 {"angle": angle_max,  "axes": (0, 2)},
                 {"angle": -angle_max, "axes": (0, 2)},
                 {"angle": angle_max,  "axes": (1, 2)},
                 {"angle": -angle_max, "axes": (1, 2)}]

    for flip in flips:
        for rot in rotations:
            tta_transforms.append({"flip": flip, "rotation": rot})
    return tta_transforms 