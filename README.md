# SWINUnetRseg

Implementace segmentace HIE lézí pomocí architektury SwinUNetR a dalších modelů.

## Popis

Tento projekt implementuje segmentaci hypoxicko-ischemické encefalopatie (HIE) lézí v MRI snímcích pomocí různých architektur hlubokých neuronových sítí, včetně SwinUNetR, AttentionUNet a UNet3Plus3D. Kód podporuje trénování na celých objemech (full-volume) i na patchích (patch-based), cross-validaci, a různé strategie inference včetně Test-Time Augmentation (TTA) a Mixture of Experts (MoE).

## Instalace

### Požadavky

- Python 3.8+
- PyTorch 1.10+
- MONAI 0.9+
- SimpleITK
- NumPy
- SciPy
- Weights & Biases (volitelné pro logování)

### Instalace balíčků

```bash
pip install -r requirements.txt
```

## Struktura projektu

```
SWINUnetRseg/
├── src/                      # Zdrojový kód
│   ├── models/               # Implementace modelů
│   │   ├── swin_unetr.py     # SwinUNetR model
│   │   ├── attention_unet.py # AttentionUNet model
│   │   └── unet3plus.py      # UNet3Plus3D model
│   ├── data/                 # Datové moduly
│   │   ├── dataset.py        # Implementace datasetů
│   │   └── preprocessing.py  # Funkce pro předzpracování dat
│   ├── loss/                 # Ztrátové funkce
│   │   └── loss_functions.py # Implementace ztrátových funkcí
│   ├── training/             # Tréninkové funkce
│   │   └── train.py          # Implementace tréninkových smyček
│   ├── inference/            # Inferenční funkce
│   │   └── inference.py      # Implementace inference
│   ├── utils/                # Pomocné funkce
│   │   └── metrics.py        # Metriky pro evaluaci
│   └── config.py             # Konfigurační modul
├── examples/                 # Příklady použití
│   ├── train_example.py      # Příklad trénování
│   └── inference_example.py  # Příklad inference
├── main.py                   # Hlavní vstupní bod
├── requirements.txt          # Seznam závislostí
└── README.md                 # Dokumentace
```

## Použití

### Trénování modelu

Pro trénování modelu můžete použít příkazovou řádku:

```bash
python main.py --mode train \
    --adc_folder /path/to/adc_data \
    --z_folder /path/to/z_adc_data \
    --label_folder /path/to/label_data \
    --model_name swinunetr \
    --training_mode full_volume \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --n_folds 5 \
    --loss_name log_cosh_dice \
    --output_dir outputs/my_training
```

Alternativně můžete upravit a spustit příklad v `examples/train_example.py`.

### Trénování s kaskádovým přístupem (specializovaný model pro malé léze)

Pro trénování modelu s kaskádovým přístupem a specializovaným modelem pro malé léze:

```bash
python main.py --mode train \
    --adc_folder /path/to/adc_data \
    --z_folder /path/to/z_adc_data \
    --label_folder /path/to/label_data \
    --model_name swinunetr \
    --in_channels 2 \
    --out_channels 2 \
    --batch_size 1 \
    --epochs 30 \
    --lr 6e-4 \
    --use_cascaded_approach \
    --small_lesion_model attention_unet \
    --small_lesion_patch_size 16 16 16 \
    --small_lesion_batch_size 128 \
    --small_lesion_max_voxels 50 \
    --small_lesion_foreground_ratio 0.8 \
    --small_lesion_large_lesion_sampling_ratio 0.25 \
    --small_lesion_epochs 15 \
    --cascaded_mode combined \
    --n_folds 5 \
    --output_dir outputs/cascaded_attention_training
```

Nový `attention_unet` model pro malé léze je optimalizován pro práci s malými patchi (16x16x16) a využívá mechanismus pozornosti pro lepší detekci detailů v malých lézích. Parametr `small_lesion_large_lesion_sampling_ratio` umožňuje redukovat počet vzorků z velkých lézí (výchozí hodnota 0.25 znamená 25% vzorků), což výrazně urychluje trénování při zachování přesnosti modelu na malých lézích.

### Inference

Pro inferenci na nových datech:

```bash
python main.py --mode inference \
    --adc_folder /path/to/adc_data \
    --z_folder /path/to/z_adc_data \
    --model_name swinunetr \
    --model_path /path/to/trained_model.pth \
    --inference_mode standard \
    --output_dir outputs/my_inference
```

Pro inferenci s Mixture of Experts (MoE):

```bash
python main.py --mode inference \
    --adc_folder /path/to/adc_data \
    --z_folder /path/to/z_adc_data \
    --model_name swinunetr \
    --model_path /path/to/main_model.pth \
    --expert_model_path /path/to/expert_model.pth \
    --inference_mode moe \
    --moe_threshold 80 \
    --output_dir outputs/my_moe_inference
```

Alternativně můžete upravit a spustit příklad v `examples/inference_example.py`.

## Konfigurace

Projekt používá konfigurační systém, který umožňuje nastavit různé parametry trénování a inference. Výchozí hodnoty jsou definovány v `src/config.py` a mohou být přepsány pomocí argumentů příkazové řádky nebo přímo v kódu.

### Hlavní konfigurační parametry

- **Obecné parametry**:

  - `mode`: Režim běhu ("train" nebo "inference")
  - `device`: Zařízení pro výpočet ("cuda" nebo "cpu")
  - `seed`: Seed pro reprodukovatelnost
  - `output_dir`: Výstupní adresář

- **Parametry datasetu**:

  - `adc_folder`: Cesta ke složce s ADC snímky
  - `z_folder`: Cesta ke složce s Z-ADC snímky
  - `label_folder`: Cesta ke složce s ground truth maskami
  - `allowed_patient_ids`: Seznam povolených ID pacientů
  - `extended_dataset`: Zda se jedná o rozšířený dataset (s aug/orig soubory)

- **Parametry modelu**:

  - `model_name`: Jméno modelu ("swinunetr", "attention_unet", "unet3plus3d")
  - `in_channels`: Počet vstupních kanálů
  - `out_channels`: Počet výstupních tříd
  - `drop_rate`: Dropout rate

- **Parametry specializovaného modelu pro malé léze**:

  - `small_lesion_model`: Jméno modelu pro detekci malých lézí ("small_unet", "simple_resunet", "attention_unet")
  - `small_lesion_patch_size`: Velikost patche pro trénování modelu malých lézí (např. [16, 16, 16])
  - `small_lesion_batch_size`: Velikost dávky pro trénování modelu malých lézí
  - `small_lesion_max_voxels`: Maximální počet voxelů pro definici malé léze
  - `small_lesion_foreground_ratio`: Poměr pozitivních vzorků při trénování modelu malých lézí
  - `small_lesion_large_lesion_sampling_ratio`: Poměr redukce vzorků z velkých lézí (0-1, výchozí 0.25)
  - `small_lesion_epochs`: Počet epoch pro trénování modelu malých lézí

- **Parametry trénování**:

  - `batch_size`: Velikost dávky
  - `epochs`: Počet epoch
  - `lr`: Learning rate
  - `training_mode`: Režim trénování ("patch" nebo "full_volume")
  - `use_augmentation`: Zda používat augmentaci
  - `n_folds`: Počet foldů pro cross-validaci

- **Parametry ztráty**:

  - `loss_name`: Jméno ztrátové funkce
  - `alpha`: Váha pro kombinované ztrátové funkce
  - `focal_alpha`, `focal_gamma`: Parametry pro Focal loss
  - `bg_weight`, `fg_weight`: Váhy pro weighted loss

- **Parametry inference**:
  - `inference_mode`: Režim inference ("standard" nebo "moe")
  - `use_tta`: Zda používat Test-Time Augmentation
  - `moe_threshold`: Threshold pro přepnutí na expertní model

## Metriky

Pro evaluaci segmentace jsou implementovány následující metriky:

- **Dice koeficient**: Měří překryv mezi predikcí a ground truth
- **Mean Average Surface Distance (MASD)**: Průměrná vzdálenost mezi povrchy segmentací
- **Normalized Surface Dice (NSD)**: Normalizovaný Dice koeficient založený na povrchových vzdálenostech

## Logování

Projekt podporuje logování pomocí Weights & Biases (wandb). Pro aktivaci logování použijte parametr `--use_wandb` a nastavte `--wandb_project` a `--wandb_run_name`.

## Licence

Tento projekt je poskytován pod licencí MIT.
