import os
import sys

# Přidání rodičovského adresáře do cesty pro import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_default_config
from main import run_cross_validation

def main():
    """
    Příklad spuštění trénování modelu.
    """
    # Získání výchozí konfigurace
    config = get_default_config()
    
    # Úprava konfigurace pro tento příklad
    config.update({
        # Cesty k datům - upravte podle vašeho umístění dat
        "adc_folder": "/path/to/adc_data",
        "z_folder": "/path/to/z_adc_data",
        "label_folder": "/path/to/label_data",
        
        # Parametry modelu
        "model_name": "swinunetr",
        "in_channels": 2,
        "out_channels": 2,
        
        # Parametry trénování
        "batch_size": 1,
        "epochs": 50,
        "lr": 1e-4,
        "training_mode": "full_volume",  # nebo "patch"
        "use_augmentation": True,
        "n_folds": 5,
        
        # Parametry ztráty
        "loss_name": "log_cosh_dice",
        
        # Výstupní adresáře
        "output_dir": "outputs/example_training",
        "model_dir": "models/example_training",
        
        # Wandb parametry
        "use_wandb": False,  # Nastavte na True, pokud chcete používat wandb
    })
    
    # Spuštění cross-validace
    run_cross_validation(config)

if __name__ == "__main__":
    main() 