import os
import numpy as np
import SimpleITK as sitk
import torch
import random
from torch.utils.data import Dataset
import re
from skimage.measure import regionprops

from .preprocessing import random_3d_augmentation, filter_augmented_files, get_base_id, heavy_3d_augmentation, soft_3d_augmentation

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
    """
    def __init__(self, adc_folder, z_folder, label_folder, augment=False, 
                 allowed_patient_ids=None, extended_dataset=True, max_aug_per_orig=0, 
                 use_z_adc=True, augmentation_type='soft'):
        super().__init__()

        self.adc_folder = adc_folder
        self.z_folder = z_folder
        self.label_folder = label_folder
        self.augment = augment
        self.extended_dataset = extended_dataset
        self.max_aug_per_orig = max_aug_per_orig
        self.use_z_adc = use_z_adc
        self.augmentation_type = augmentation_type

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

        adc_np   = sitk.GetArrayFromImage(sitk.ReadImage(adc_path))
        label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        # Načtení Z-ADC mapy pouze pokud se používá
        if self.use_z_adc and self.z_folder:
            zadc_path = os.path.join(self.z_folder, self.z_files[idx])
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


def get_subject_id_from_filename(filename):
    """
    Extrahuje ID subjektu z názvu souboru.
    
    Args:
        filename: Název souboru k analýze
        
    Returns:
        str: ID subjektu nebo None, pokud nelze extrahovat
    """
    # Odstranění přípony souboru
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Odstranění případných "ADC_", "LABEL_", "Z_ADC_" prefixů
    for prefix in ["ADC_", "LABEL_", "Z_ADC_", "adc_", "label_", "z_adc_"]:
        if name_without_ext.startswith(prefix):
            name_without_ext = name_without_ext[len(prefix):]
            
    # Odstranění případných "_aug" sufixů
    if "_aug" in name_without_ext.lower():
        parts = name_without_ext.lower().split("_aug")
        name_without_ext = parts[0]
    
    return name_without_ext


def extract_patient_id(filepath):
    """
    Extrahuje ID pacienta z cesty k souboru.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    return None


class SmallLesionPatchDataset(Dataset):
    """
    Dataset pro extrakci malých patchů ze 3D objemů, zaměřený především na oblasti s malými lézemi.
    
    Tento dataset je navržen speciálně pro trénování modelu malých lézí, který potřebuje 
    zpracovávat menší patche a lépe se soustředit na malé léze.
    """
    def __init__(
        self,
        adc_folder,
        z_folder,
        label_folder,
        patch_size=(16, 16, 16),
        patches_per_volume=200,
        foreground_ratio=0.8,
        small_lesion_max_voxels=50,
        augment=True,
        use_z_adc=True,
        seed=42,
        specific_files=None,
        large_lesion_sampling_ratio=0.25  # Nový parametr pro kontrolu vzorkování velkých lézí
    ):
        """
        Args:
            adc_folder: Cesta ke složce s ADC snímky
            z_folder: Cesta ke složce s Z-ADC snímky
            label_folder: Cesta ke složce s ground truth maskami
            patch_size: Velikost extrahovaných patchů [D, H, W]
            patches_per_volume: Počet patchů extrahovaných z každého objemu
            foreground_ratio: Poměr patchů, které musí obsahovat lézi (0-1)
            small_lesion_max_voxels: Maximální počet voxelů pro klasifikaci léze jako "malé"
            augment: Zda provádět augmentaci dat
            use_z_adc: Zda používat Z-ADC snímky jako druhý kanál
            seed: Seed pro reprodukovatelnost
            specific_files: Slovník obsahující seznamy konkrétních souborů, které mají být použity
                           {'adc_files': [...], 'z_files': [...], 'lab_files': [...]}
            large_lesion_sampling_ratio: Poměr redukce vzorků z velkých lézí (0-1, výchozí 0.25)
        """
        super().__init__()
        self.adc_folder = adc_folder
        self.z_folder = z_folder
        self.label_folder = label_folder
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.foreground_ratio = foreground_ratio
        self.small_lesion_max_voxels = small_lesion_max_voxels
        self.augment = augment
        self.use_z_adc = use_z_adc
        self.seed = seed
        self.large_lesion_sampling_ratio = large_lesion_sampling_ratio
        
        # Nastavení seedu pro reprodukovatelnost
        np.random.seed(seed)
        
        # Načtení seznamu souborů
        if specific_files is None:
            # Standardní načtení všech souborů
            self.adc_files = sorted([f for f in os.listdir(adc_folder) if f.endswith(('.mha', '.nii', '.nii.gz'))])
            self.lab_files = sorted([f for f in os.listdir(label_folder) if f.endswith(('.mha', '.nii', '.nii.gz'))])
            if use_z_adc:
                self.z_files = sorted([f for f in os.listdir(z_folder) if f.endswith(('.mha', '.nii', '.nii.gz'))])
        else:
            # Použití konkrétních souborů
            self.adc_files = specific_files['adc_files']
            self.lab_files = specific_files['lab_files']
            if use_z_adc:
                self.z_files = specific_files['z_files']
        
        # Kontrola, zda počty souborů souhlasí
        assert len(self.adc_files) == len(self.lab_files), "Počet ADC a label souborů nesouhlasí"
        if use_z_adc:
            assert len(self.adc_files) == len(self.z_files), "Počet ADC a Z-ADC souborů nesouhlasí"
        
        # Analýza všech objemů pro identifikaci malých lézí
        self.volume_info = self._analyze_volumes()
        
        # Vytvoření seznamu všech dostupných patchů
        self.all_patches = self._create_patch_list()
        
        # Výpis statistik o počtu patchů
        small_lesion_volumes_count = len([info for info in self.volume_info if info['is_small_lesion'] and info['has_lesions']])
        large_lesion_volumes_count = len([info for info in self.volume_info if not info['is_small_lesion'] and info['has_lesions']])
        no_lesion_volumes_count = len([info for info in self.volume_info if not info['has_lesions']])
        
        print(f"\nStatistika vzorků pro malý model:")
        print(f"  Celkový počet vzorků: {len(self.all_patches)}")
        print(f"  Objemy s malými lézemi: {small_lesion_volumes_count}")
        print(f"  Objemy s velkými lézemi: {large_lesion_volumes_count}")
        print(f"  Objemy bez lézí: {no_lesion_volumes_count}")
        print(f"  Vzorky z volimů s malými lézemi: ~{small_lesion_volumes_count * self.patches_per_volume} (zachováno 100%)")
        print(f"  Vzorky z volimů s velkými lézemi: ~{int(large_lesion_volumes_count * self.patches_per_volume * self.large_lesion_sampling_ratio)} (redukováno na {self.large_lesion_sampling_ratio*100:.0f}%)")
        print(f"  Vzorky z objemů bez lézí: ~{min(no_lesion_volumes_count * 50, len(self.all_patches) - small_lesion_volumes_count * self.patches_per_volume - int(large_lesion_volumes_count * self.patches_per_volume * self.large_lesion_sampling_ratio))}")
        
    def _analyze_volumes(self):
        """
        Analyzuje všechny objemy a klasifikuje léze podle velikosti.
        
        Returns:
            list: Seznam informací o objemech a klasifikace lézí
        """
        volume_info = []
        
        for i, (adc_file, lab_file) in enumerate(zip(self.adc_files, self.lab_files)):
            lab_path = os.path.join(self.label_folder, lab_file)
            lab_sitk = sitk.ReadImage(lab_path)
            lab_np = sitk.GetArrayFromImage(lab_sitk)
            
            # Zjištění, zda objem obsahuje léze
            has_lesions = np.max(lab_np) > 0
            
            if has_lesions:
                # Počet foreground voxelů
                lesion_voxels = np.sum(lab_np > 0)
                
                # Klasifikace léze jako malé nebo velké
                is_small_lesion = lesion_voxels <= self.small_lesion_max_voxels
                
                # Připravíme masky pro možné hledání patchů
                props = regionprops(lab_np.astype(np.int32))
                
                # Sbíráme souřadnice středů lézí pro inteligentní vzorkování
                centers = []
                for prop in props:
                    centers.append(prop.centroid)
                
                volume_info.append({
                    'index': i,
                    'adc_file': adc_file,
                    'lab_file': lab_file,
                    'has_lesions': has_lesions,
                    'lesion_voxels': lesion_voxels,
                    'is_small_lesion': is_small_lesion,
                    'lesion_centers': centers
                })
            else:
                volume_info.append({
                    'index': i,
                    'adc_file': adc_file,
                    'lab_file': lab_file,
                    'has_lesions': has_lesions,
                    'lesion_voxels': 0,
                    'is_small_lesion': False,
                    'lesion_centers': []
                })
        
        return volume_info
    
    def _create_patch_list(self):
        """
        Vytvoří seznam všech patchů, které budou použity pro trénování.
        Prioritizuje patche obsahující malé léze.
        
        Returns:
            list: Seznam informací o patchích [(volume_idx, coord_z, coord_y, coord_x), ...]
        """
        all_patches = []
        
        # Prioritní výběr patchů z objemů s malými lézemi - tyto zachováváme všechny
        small_lesion_volumes = [info for info in self.volume_info if info['is_small_lesion'] and info['has_lesions']]
        large_lesion_volumes = [info for info in self.volume_info if not info['is_small_lesion'] and info['has_lesions']]
        no_lesion_volumes = [info for info in self.volume_info if not info['has_lesions']]
        
        print(f"Zpracovávám {len(small_lesion_volumes)} objemů s malými lézemi...")
        
        # Nejprve zpracujeme objemy s malými lézemi - zachováváme všechny
        for vol_info in small_lesion_volumes:
            vol_idx = vol_info['index']
            
            # Načtení label dat
            lab_path = os.path.join(self.label_folder, vol_info['lab_file'])
            lab_sitk = sitk.ReadImage(lab_path)
            lab_np = sitk.GetArrayFromImage(lab_sitk)
            
            # Načtení předpočítaných středů lézí
            centers = vol_info['lesion_centers']
            
            # Výběr náhodných patchů se zaměřením na oblasti lézí - zachováváme plný počet
            vol_patches = self._sample_patches_from_volume(
                vol_idx, lab_np, centers, self.patches_per_volume
            )
            all_patches.extend(vol_patches)
        
        print(f"Zpracovávám {len(large_lesion_volumes)} objemů s velkými lézemi (redukovaný počet vzorků)...")
        
        # Zpracování objemů s velkými lézemi - redukovaný počet vzorků
        if large_lesion_volumes:
            # Redukujeme počet patchů z velkých lézí
            patches_per_large_volume = int(self.patches_per_volume * self.large_lesion_sampling_ratio)
            
            for vol_info in large_lesion_volumes:
                vol_idx = vol_info['index']
                
                # Načtení label dat
                lab_path = os.path.join(self.label_folder, vol_info['lab_file'])
                lab_sitk = sitk.ReadImage(lab_path)
                lab_np = sitk.GetArrayFromImage(lab_sitk)
                
                # Načtení předpočítaných středů lézí
                centers = vol_info['lesion_centers']
                
                # Výběr redukovaného počtu náhodných patchů
                vol_patches = self._sample_patches_from_volume(
                    vol_idx, lab_np, centers, patches_per_large_volume
                )
                all_patches.extend(vol_patches)
        
        print(f"Zpracovávám {len(no_lesion_volumes)} objemů bez lézí (negativní příklady)...")
        
        # Přidání omezeného počtu vzorků z objemů bez lézí (negativní příklady)
        if no_lesion_volumes:
            # Maximálně 50 patchů z každého objemu bez lézí
            patches_per_empty_volume = min(50, self.patches_per_volume // 4)
            
            for vol_info in no_lesion_volumes:
                vol_idx = vol_info['index']
                
                # Načtení label dat
                lab_path = os.path.join(self.label_folder, vol_info['lab_file'])
                lab_sitk = sitk.ReadImage(lab_path)
                lab_np = sitk.GetArrayFromImage(lab_sitk)
                
                # Výběr omezeného počtu náhodných patchů
                vol_patches = self._sample_patches_from_volume(
                    vol_idx, lab_np, [], patches_per_empty_volume  # prázdný seznam středů = náhodné vzorkování
                )
                all_patches.extend(vol_patches)
        
        # Náhodně promícháme patche
        np.random.shuffle(all_patches)
        
        print(f"Celkem vygenerováno {len(all_patches)} patchů")
        
        return all_patches
    
    def _sample_patches_from_volume(self, vol_idx, label_data, lesion_centers, num_patches):
        """
        Vzorkuje patche z daného objemu s důrazem na malé léze.
        
        Args:
            vol_idx: Index objemu v datasetu
            label_data: Numpy array s ground truth daty
            lesion_centers: Seznam souřadnic středů lézí
            num_patches: Počet patchů k výběru
            
        Returns:
            list: Seznam informací o patchích [(vol_idx, z, y, x), ...]
        """
        d, h, w = label_data.shape
        pz, py, px = self.patch_size
        
        patches = []
        
        # Určení poměru foreground vs. background patchů
        num_fg_patches = int(num_patches * self.foreground_ratio)
        num_bg_patches = num_patches - num_fg_patches
        
        # --- Foreground patche ---
        if lesion_centers and num_fg_patches > 0:
            # Máme léze, vzorkujeme z jejich blízkosti
            for _ in range(num_fg_patches):
                # Náhodný výběr centra léze
                center = lesion_centers[np.random.randint(0, len(lesion_centers))]
                cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
                
                # Náhodný offset od středu (+-8 voxelů)
                offset_z = np.random.randint(-4, 5) if pz > 8 else 0
                offset_y = np.random.randint(-4, 5) if py > 8 else 0
                offset_x = np.random.randint(-4, 5) if px > 8 else 0
                
                # Výpočet středu patche
                z = max(pz // 2, min(d - pz // 2, cz + offset_z))
                y = max(py // 2, min(h - py // 2, cy + offset_y))
                x = max(px // 2, min(w - px // 2, cx + offset_x))
                
                # Výpočet souřadnic okraje patche
                z_start = z - pz // 2
                y_start = y - py // 2
                x_start = x - px // 2
                
                patches.append((vol_idx, z_start, y_start, x_start))
        else:
            # Náhodné foreground patche, pokud nemáme léze
            fg_indices = np.where(label_data > 0)
            if len(fg_indices[0]) > 0:
                for _ in range(num_fg_patches):
                    # Náhodný výběr foreground voxelu
                    idx = np.random.randint(0, len(fg_indices[0]))
                    z_fg, y_fg, x_fg = fg_indices[0][idx], fg_indices[1][idx], fg_indices[2][idx]
                    
                    # Výpočet středu patche
                    z = max(pz // 2, min(d - pz // 2, z_fg))
                    y = max(py // 2, min(h - py // 2, y_fg))
                    x = max(px // 2, min(w - px // 2, x_fg))
                    
                    # Výpočet souřadnic okraje patche
                    z_start = z - pz // 2
                    y_start = y - py // 2
                    x_start = x - px // 2
                    
                    patches.append((vol_idx, z_start, y_start, x_start))
            else:
                # Pokud nemáme foreground voxely, přidáme náhodné patche
                for _ in range(num_fg_patches):
                    z_start = np.random.randint(0, d - pz + 1)
                    y_start = np.random.randint(0, h - py + 1)
                    x_start = np.random.randint(0, w - px + 1)
                    
                    patches.append((vol_idx, z_start, y_start, x_start))
        
        # --- Background patche ---
        for _ in range(num_bg_patches):
            z_start = np.random.randint(0, d - pz + 1)
            y_start = np.random.randint(0, h - py + 1)
            x_start = np.random.randint(0, w - px + 1)
            
            patches.append((vol_idx, z_start, y_start, x_start))
        
        return patches
    
    def __len__(self):
        return len(self.all_patches)
    
    def __getitem__(self, idx):
        # Získání informací o patchi
        vol_idx, z_start, y_start, x_start = self.all_patches[idx]
        pz, py, px = self.patch_size
        
        # Načtení ADC dat
        adc_path = os.path.join(self.adc_folder, self.adc_files[vol_idx])
        adc_sitk = sitk.ReadImage(adc_path)
        adc_np = sitk.GetArrayFromImage(adc_sitk).astype(np.float32)
        
        # Extrakce ADC patche
        adc_patch = adc_np[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px].copy()
        
        # Načtení a extrakce Z-ADC patche, pokud je požadován
        if self.use_z_adc:
            z_path = os.path.join(self.z_folder, self.z_files[vol_idx])
            z_sitk = sitk.ReadImage(z_path)
            z_np = sitk.GetArrayFromImage(z_sitk).astype(np.float32)
            
            # Extrakce Z-ADC patche
            z_patch = z_np[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px].copy()
            
            # Sloučení kanálů
            input_data = np.stack([adc_patch, z_patch], axis=0)
        else:
            # Pouze ADC
            input_data = np.expand_dims(adc_patch, axis=0)
        
        # Načtení a extrakce label patche
        lab_path = os.path.join(self.label_folder, self.lab_files[vol_idx])
        lab_sitk = sitk.ReadImage(lab_path)
        lab_np = sitk.GetArrayFromImage(lab_sitk)
        
        # Extrakce label patche
        lab_patch = lab_np[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px].copy()
        
        # Převod na tensor
        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(lab_patch).long()
        
        return input_tensor, label_tensor 