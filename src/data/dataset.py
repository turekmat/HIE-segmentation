import os
import numpy as np
import SimpleITK as sitk
import torch
import random
from torch.utils.data import Dataset
import re

from .preprocessing import random_3d_augmentation, filter_augmented_files, get_base_id, heavy_3d_augmentation, soft_3d_augmentation, select_inpainted_data_for_training

class BONBID3DFullVolumeDataset(Dataset):
    """
    Čte 3D volume (ADC, Z_ADC, LABEL) z daných složek. Lze omezit dataset pouze
    na subjekty s patient ID z allowed_patient_ids.
    
    Parametry:
        adc_folder: Složka s ADC soubory
        z_folder: Složka s Z-ADC soubory (může být None, pokud use_z_adc=False)
        label_folder: Složka s label soubory
        augment: Zda použít augmentaci při načítání
        allowed_patient_ids: Seznam ID pacientů, kteří mají být v datasetu (None = všichni)
        extended_dataset: Zda se jedná o rozšířený dataset s orig/aug soubory
        max_aug_per_orig: Maximální počet augmentovaných souborů na jeden originální
        use_z_adc: Zda používat Z-ADC modalitu (pokud False, použije se pouze ADC)
        augmentation_type: Typ augmentace ('soft' nebo 'heavy')
        use_inpainted_lesions: Zda používat inpaintnuté léze během tréninku
        inpaint_adc_folder: Složka s inpaintnutými ADC soubory
        inpaint_z_folder: Složka s inpaintnutými Z-ADC soubory
        inpaint_label_folder: Složka s inpaintnutými LABEL soubory
        inpaint_probability: Pravděpodobnost použití inpaintnutých dat (0.0-1.0)
    """
    def __init__(self, adc_folder, z_folder, label_folder, augment=False, 
                 allowed_patient_ids=None, extended_dataset=True, max_aug_per_orig=0, 
                 use_z_adc=True, augmentation_type='soft',
                 use_inpainted_lesions=False, inpaint_adc_folder=None, 
                 inpaint_z_folder=None, inpaint_label_folder=None,
                 inpaint_probability=0.2):
        super().__init__()

        self.adc_folder = adc_folder
        self.z_folder = z_folder
        self.label_folder = label_folder
        self.augment = augment
        self.extended_dataset = extended_dataset
        self.max_aug_per_orig = max_aug_per_orig
        self.use_z_adc = use_z_adc
        self.augmentation_type = augmentation_type
        
        # Nastavení pro inpaintnuté léze
        self.use_inpainted_lesions = use_inpainted_lesions
        self.inpaint_adc_folder = inpaint_adc_folder
        self.inpaint_z_folder = inpaint_z_folder
        self.inpaint_label_folder = inpaint_label_folder
        self.inpaint_probability = inpaint_probability
        
        # Kontrola správného nastavení parametrů pro inpaintnuté léze
        if use_inpainted_lesions and (not inpaint_adc_folder or not inpaint_label_folder):
            print("Warning: Inpainted lesions enabled but folders not provided. Disabling inpainted lesions.")
            self.use_inpainted_lesions = False
        
        if use_inpainted_lesions and self.use_z_adc and not inpaint_z_folder:
            print("Warning: Z-ADC is enabled but inpaint_z_folder not provided. Disabling inpainted lesions.")
            self.use_inpainted_lesions = False

        self.adc_files = sorted([f for f in os.listdir(adc_folder) if f.endswith('.mha')])
        
        if z_folder and use_z_adc:
            self.z_files = sorted([f for f in os.listdir(z_folder) if f.endswith('.mha')])
        else:
            # Pokud nepoužíváme Z-ADC, vytvoříme kopii seznamu ADC souborů pro zachování kompatibility
            self.z_files = self.adc_files.copy()
            
        self.lab_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mha')])

        if extended_dataset:
            self.adc_files = filter_augmented_files(self.adc_files, max_aug_per_orig)
            self.z_files   = filter_augmented_files(self.z_files, max_aug_per_orig)
            self.lab_files = filter_augmented_files(self.lab_files, max_aug_per_orig)
            print(f"DEBUG DATASET: Po filtrování -> ADC: {len(self.adc_files)} souborů, Z_ADC: {len(self.z_files)} souborů, LABEL: {len(self.lab_files)} souborů")

        if allowed_patient_ids is not None:
            self.adc_files = [f for f in self.adc_files if self.get_patient_numeric_id(f) in allowed_patient_ids]
            self.z_files   = [f for f in self.z_files if self.get_patient_numeric_id(f) in allowed_patient_ids]
            self.lab_files = [f for f in self.lab_files if self.get_patient_numeric_id(f) in allowed_patient_ids]

        if not (len(self.adc_files) == len(self.z_files) == len(self.lab_files)):
            raise ValueError("Mismatch in .mha file counts among ADC, Z_ADC, LABEL.")

    def get_patient_numeric_id(self, filename):
        """
        Extrahuje numerické ID pacienta z názvu souboru.
        """
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return -1

    def __len__(self):
        return len(self.adc_files)

    def __getitem__(self, idx):
        adc_path   = os.path.join(self.adc_folder, self.adc_files[idx])
        label_path = os.path.join(self.label_folder, self.lab_files[idx])
        
        if self.use_z_adc and self.z_folder:
            zadc_path = os.path.join(self.z_folder, self.z_files[idx])
        else:
            zadc_path = None
        
        # Možnost výběru inpaintnutých dat s danou pravděpodobností
        if self.use_inpainted_lesions:
            adc_path, zadc_path, label_path = select_inpainted_data_for_training(
                adc_path, zadc_path, label_path,
                self.inpaint_adc_folder, self.inpaint_z_folder, self.inpaint_label_folder,
                self.inpaint_probability
            )

        # Načtení vybraných dat
        adc_np   = sitk.GetArrayFromImage(sitk.ReadImage(adc_path))
        label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        # Načtení Z-ADC mapy pouze pokud se používá
        if self.use_z_adc and zadc_path:
            zadc_np  = sitk.GetArrayFromImage(sitk.ReadImage(zadc_path))
        else:
            # Vytvoříme dummy tensor nulový, který nahradíme později (nebude použit)
            zadc_np = np.zeros_like(adc_np)

        if self.augment:
            if self.augmentation_type == 'heavy':
                adc_np, zadc_np, label_np = heavy_3d_augmentation(adc_np, zadc_np, label_np)
            else:
                # Výchozí je soft augmentace
                adc_np, zadc_np, label_np = soft_3d_augmentation(adc_np, zadc_np, label_np)

        # Převedeme na Torch tensor
        adc_t  = torch.from_numpy(adc_np).unsqueeze(0)
        
        # Pokud používáme Z-ADC, vytvoříme vstup s 2 kanály, jinak pouze s 1 kanálem
        if self.use_z_adc:
            zadc_t = torch.from_numpy(zadc_np).unsqueeze(0)
            input_t = torch.cat([adc_t, zadc_t], dim=0)  # (2, D, H, W)
        else:
            input_t = adc_t  # (1, D, H, W)
            
        label_t = torch.from_numpy(label_np).long()

        return input_t.float(), label_t


class IndexedDatasetWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)


class BONBID3DPatchDataset(Dataset):
    """
    Třída, která obalí full-volume dataset a při každém volání __getitem__
    náhodně vybere patch z daného objemu.

    Args:
        full_volume_dataset: Instance již existujícího full-volume datasetu (např. BONBID3DFullVolumeDataset nebo jeho Subset).
        patch_size (tuple): Velikost patche, např. (64,64,64).
        patches_per_volume (int): Kolik patchí se bude extrahovat z každého objemu.
        augment (bool): Použít nebo ne data augmentaci na jednotlivých patchech.
        intelligent_sampling (bool): Použít inteligentní vzorkování zaměřené na oblasti s lézemi.
        foreground_ratio (float): Poměr patchů, které by měly obsahovat popředí (léze).
        augmentation_type (str): Typ augmentace ('soft' nebo 'heavy').
    """
    def __init__(self, full_volume_dataset, patch_size=(64,64,64), patches_per_volume=10, 
                 augment=False, intelligent_sampling=True, foreground_ratio=0.7,
                 augmentation_type='soft'):
        super().__init__()
        self.full_volume_dataset = full_volume_dataset
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.num_volumes = len(full_volume_dataset)
        self.intelligent_sampling = intelligent_sampling
        self.foreground_ratio = foreground_ratio
        self.augmentation_type = augmentation_type
        self.label_centers = {}  # Cache pro středy lézí

    def __len__(self):
        return self.num_volumes * self.patches_per_volume

    def __getitem__(self, idx):
        # Určíme, z kterého objemu bude patch
        volume_idx = idx // self.patches_per_volume
        input_vol, label_vol = self.full_volume_dataset[volume_idx]  # input_vol: (C, D, H, W), label_vol: (D, H, W)

        # Převedeme na numpy (pro snadné ořezy)
        input_np = input_vol.numpy()  # tvar (C, D, H, W)
        label_np = label_vol.numpy()  # tvar (D, H, W)

        C, D, H, W = input_np.shape
        pD, pH, pW = self.patch_size

        # Vytvoříme paddovaný tensor, pokud objem je menší než požadovaná velikost patche
        if D < pD or H < pH or W < pW:
            # Vytvoříme paddovaný array pro vstup
            padded_input = np.zeros((C, max(D, pD), max(H, pH), max(W, pW)), dtype=input_np.dtype)
            padded_input[:, :D, :H, :W] = input_np
            input_np = padded_input
            
            # Vytvoříme paddovaný array pro label
            padded_label = np.zeros((max(D, pD), max(H, pH), max(W, pW)), dtype=label_np.dtype)
            padded_label[:D, :H, :W] = label_np
            label_np = padded_label
            
            # Aktualizujeme rozměry
            _, D, H, W = input_np.shape

        # Inteligentní vzorkování - zaměřujeme se na oblasti s lézemi
        if self.intelligent_sampling and np.random.random() < self.foreground_ratio:
            # Použít cache pokud máme již spočítané středy lézí pro tento objem
            if volume_idx in self.label_centers:
                foreground_centers = self.label_centers[volume_idx]
            else:
                # Najdeme indexy voxelů, které obsahují léze (hodnoty > 0)
                foreground_indices = np.where(label_np > 0)
                if len(foreground_indices[0]) > 0:
                    # Náhodně vybereme jeden index léze jako centrum patche
                    idx_choice = np.random.randint(0, len(foreground_indices[0]))
                    foreground_centers = [
                        (foreground_indices[0][idx_choice], 
                         foreground_indices[1][idx_choice],
                         foreground_indices[2][idx_choice])
                    ]
                else:
                    foreground_centers = []
                self.label_centers[volume_idx] = foreground_centers
            
            if foreground_centers:
                # Vybereme náhodné centrum z foreground_centers
                center = random.choice(foreground_centers)
                # Určíme hranice patche tak, aby centrum bylo někde v patchi
                half_pD, half_pH, half_pW = pD // 2, pH // 2, pW // 2
                d_offset = np.random.randint(-half_pD, half_pD)
                h_offset = np.random.randint(-half_pH, half_pH)
                w_offset = np.random.randint(-half_pW, half_pW)
                
                d0 = max(0, min(D - pD, center[0] - half_pD + d_offset))
                h0 = max(0, min(H - pH, center[1] - half_pH + h_offset))
                w0 = max(0, min(W - pW, center[2] - half_pW + w_offset))
            else:
                # Pokud nejsou léze, použijeme náhodné vzorkování
                d0 = random.randint(0, max(0, D - pD))
                h0 = random.randint(0, max(0, H - pH))
                w0 = random.randint(0, max(0, W - pW))
        else:
            # Standardní náhodné vzorkování
            d0 = random.randint(0, max(0, D - pD))
            h0 = random.randint(0, max(0, H - pH))
            w0 = random.randint(0, max(0, W - pW))

        # Ořízneme patch z každého kanálu a labelu
        patch_input = input_np[:, d0:d0+pD, h0:h0+pH, w0:w0+pW]
        patch_label = label_np[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        
        # Zajistíme, že patch má přesně požadovanou velikost (pro případ, že by padding nepracoval správně)
        if patch_input.shape[1:] != (pD, pH, pW):
            temp_input = np.zeros((C, pD, pH, pW), dtype=patch_input.dtype)
            temp_input[:, :min(pD, patch_input.shape[1]), :min(pH, patch_input.shape[2]), :min(pW, patch_input.shape[3])] = \
                patch_input[:, :min(pD, patch_input.shape[1]), :min(pH, patch_input.shape[2]), :min(pW, patch_input.shape[3])]
            patch_input = temp_input
            
            temp_label = np.zeros((pD, pH, pW), dtype=patch_label.dtype)
            temp_label[:min(pD, patch_label.shape[0]), :min(pH, patch_label.shape[1]), :min(pW, patch_label.shape[2])] = \
                patch_label[:min(pD, patch_label.shape[0]), :min(pH, patch_label.shape[1]), :min(pW, patch_label.shape[2])]
            patch_label = temp_label

        # Pokud chcete augmentaci na patchi, použijte ji zde.
        if self.augment:
            # Uvažujeme, že první kanál je ADC a druhý je Z_ADC – zavoláme tedy existující funkci
            adc_patch = patch_input[0]
            zadc_patch = patch_input[1] if C > 1 else np.zeros_like(adc_patch)
            
            if self.augmentation_type == 'heavy':
                adc_patch, zadc_patch, patch_label = heavy_3d_augmentation(adc_patch, zadc_patch, patch_label)
            else:
                # Výchozí je soft augmentace
                adc_patch, zadc_patch, patch_label = soft_3d_augmentation(adc_patch, zadc_patch, patch_label)
                
            patch_input = np.stack([adc_patch, zadc_patch], axis=0) if C > 1 else np.expand_dims(adc_patch, axis=0)

        # Převedeme zpět na Torch tensory
        patch_input = torch.from_numpy(patch_input).float()
        patch_label = torch.from_numpy(patch_label).long()
        
        # Konečná kontrola tvaru
        assert patch_input.shape[1:] == (pD, pH, pW), f"Nesprávná velikost patch_input: {patch_input.shape}, očekáváno (C, {pD}, {pH}, {pW})"
        assert patch_label.shape == (pD, pH, pW), f"Nesprávná velikost patch_label: {patch_label.shape}, očekáváno ({pD}, {pH}, {pW})"

        return patch_input, patch_label


def get_subject_id_from_filename(filename: str):
    """
    Extrahuje ID subjektu (pacient) z názvu souboru.
    Podporuje různé formáty názvů souborů.
    """
    # Zkus najít ID s podtržítkem před a za číslem (např. _123_)
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    
    # Pokus najít ID na začátku s podtržítkem za ním (např. 123_)
    match = re.search(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    
    # Pokus najít ID pomocí get_base_id
    base_id = get_base_id(filename)
    if base_id:
        match = re.search(r'(\d+)', base_id)
        if match:
            return match.group(1)
    
    # Poslední pokus - jakékoliv číslo v názvu
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    
    # Pokud nic nenajdeme, vrátíme celý název souboru
    return filename


def extract_patient_id(filepath):
    """
    Extrahuje ID pacienta z cesty k souboru.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    return None 