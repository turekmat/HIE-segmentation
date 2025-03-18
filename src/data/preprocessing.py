import os
import numpy as np
import SimpleITK as sitk
import random
import re
from scipy.ndimage import rotate, gaussian_filter, distance_transform_edt, binary_erosion

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
    Provádí normalizaci intensit a volitelně normalizaci prostorových kroků.

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
    # Kontrola, zda výstupní adresáře existují, případně je vytvoří
    if not os.path.exists(output_adc):
        os.makedirs(output_adc)
    if not os.path.exists(output_z):
        os.makedirs(output_z)
    if not os.path.exists(output_label):
        os.makedirs(output_label)

    # Kontrola, zda již existují předzpracované soubory
    existing_adc_mhas = [f for f in os.listdir(output_adc) if f.endswith('.mha')]
    existing_z_mhas   = [f for f in os.listdir(output_z)   if f.endswith('.mha')]
    existing_label_mhas = [f for f in os.listdir(output_label) if f.endswith('.mha')]

    if (
        len(existing_adc_mhas) > 0 and
        len(existing_adc_mhas) == len(existing_z_mhas) == len(existing_label_mhas)
    ):
        print("Preprocessed data already exist. Skipping preprocessing.")
        return

    # Jinak provést předzpracování
    print("No (or incomplete) preprocessed data found. Starting preprocessing...")

    adc_files   = sorted([f for f in os.listdir(adc_folder) if f.endswith('.mha')])
    z_files     = sorted([f for f in os.listdir(z_folder)   if f.endswith('.mha')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mha')])

    if not (len(adc_files) == len(z_files) == len(label_files)):
        raise ValueError("Inconsistent number of input files across folders.")

    # Zpracujte každý sken a anotaci
    for i, (adc_file, z_file, label_file) in enumerate(zip(adc_files, z_files, label_files)):
        print(f"Processing {i+1}/{len(adc_files)}: {adc_file}")

        # Načtení vstupních skenů
        adc_path = os.path.join(adc_folder, adc_file)
        z_path = os.path.join(z_folder, z_file)
        label_path = os.path.join(label_folder, label_file)

        adc_sitk = sitk.ReadImage(adc_path)
        z_sitk = sitk.ReadImage(z_path)
        label_sitk = sitk.ReadImage(label_path)

        # Normalizace spacing (volitelné)
        if allow_normalize_spacing:
            adc_sitk = normalize_spacing(adc_sitk)
            z_sitk = normalize_spacing(z_sitk)
            label_sitk = normalize_spacing(label_sitk)

        # Převod na numpy pro zpracování
        adc_np = sitk.GetArrayFromImage(adc_sitk)
        z_np = sitk.GetArrayFromImage(z_sitk)
        label_np = sitk.GetArrayFromImage(label_sitk)

        # Oříznutí na oblast zájmu a padding na násobky 32
        bbox = compute_largest_3d_bounding_box([adc_np, z_np, label_np], threshold=0)
        adc_np, z_np, label_np = crop_to_largest_bounding_box(adc_np, z_np, label_np, bbox, margin=5)

        adc_np = pad_3d_all_dims_to_multiple_of_32(adc_np)
        z_np = pad_3d_all_dims_to_multiple_of_32(z_np)
        label_np = pad_3d_all_dims_to_multiple_of_32(label_np)

        # Normalizace intenzit (volitelné)
        if normalize:
            adc_np = z_score_normalize(adc_np)
            z_np = z_score_normalize(z_np)

        # Převod zpět na SimpleITK obrazy
        processed_adc = sitk.GetImageFromArray(adc_np)
        processed_z = sitk.GetImageFromArray(z_np)
        processed_label = sitk.GetImageFromArray(label_np)

        # Zkopírování metadat
        processed_adc.CopyInformation(adc_sitk)
        processed_z.CopyInformation(z_sitk)
        processed_label.CopyInformation(label_sitk)

        # Uložení předzpracovaných souborů
        sitk.WriteImage(processed_adc, os.path.join(output_adc, adc_file))
        sitk.WriteImage(processed_z, os.path.join(output_z, z_file))
        sitk.WriteImage(processed_label, os.path.join(output_label, label_file))

    print("Preprocessing complete.")


def compute_largest_3d_bounding_box(volumes, threshold=0):
    """
    Vypočítá největší bounding box ze všech vstupních objemů.
    """
    min_d, min_h, min_w = float('inf'), float('inf'), float('inf')
    max_d, max_h, max_w = 0, 0, 0

    for vol in volumes:
        non_zero = np.where(vol > threshold)
        if len(non_zero[0]) > 0:
            min_d = min(min_d, np.min(non_zero[0]))
            min_h = min(min_h, np.min(non_zero[1]))
            min_w = min(min_w, np.min(non_zero[2]))
            max_d = max(max_d, np.max(non_zero[0]))
            max_h = max(max_h, np.max(non_zero[1]))
            max_w = max(max_w, np.max(non_zero[2]))

    return (min_d, max_d, min_h, max_h, min_w, max_w)


def crop_to_largest_bounding_box(adc_np, zadc_np, label_np, bounding_box, margin=5):
    """
    Ořízne objemy podle bounding boxu s přidaným okrajem.
    """
    min_d, max_d, min_h, max_h, min_w, max_w = bounding_box
    D, H, W = adc_np.shape

    # Přidání okraje
    min_d = max(0, min_d - margin)
    min_h = max(0, min_h - margin)
    min_w = max(0, min_w - margin)
    max_d = min(D - 1, max_d + margin)
    max_h = min(H - 1, max_h + margin)
    max_w = min(W - 1, max_w + margin)

    # Oříznutí objemů
    adc_cropped = adc_np[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
    zadc_cropped = zadc_np[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
    label_cropped = label_np[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]

    return adc_cropped, zadc_cropped, label_cropped


def pad_3d_all_dims_to_multiple_of_32(volume_3d, mode="edge"):
    """
    Přidá padding k 3D objemu tak, aby všechny rozměry byly násobky 32.
    """
    D, H, W = volume_3d.shape

    def pad_dim_to_32(dim_size):
        if dim_size % 32 == 0:
            return dim_size
        return ((dim_size - 1) // 32 + 1) * 32

    newD = pad_dim_to_32(D)
    newH = pad_dim_to_32(H)
    newW = pad_dim_to_32(W)

    padD = newD - D
    padH = newH - H
    padW = newW - W

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

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

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
    """
    if mask is not None:
        values = image_np[mask > 0]
        if len(values) == 0:  # Prázdná maska
            mean_val = np.mean(image_np)
            std_val = np.std(image_np)
        else:
            mean_val = np.mean(values)
            std_val = np.std(values)
    else:
        mean_val = np.mean(image_np)
        std_val = np.std(image_np)

    # Prevence dělení nulou
    if std_val < 1e-10:
        std_val = 1.0

    normalized = (image_np - mean_val) / std_val
    return normalized


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