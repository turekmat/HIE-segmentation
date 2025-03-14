import os
import sys

# Přidání rodičovského adresáře do cesty pro import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_default_config
from main import run_inference

def main():
    """
    Příklad spuštění inference modelu.
    """
    # Získání výchozí konfigurace
    config = get_default_config()
    
    # Úprava konfigurace pro tento příklad
    config.update({
        # Režim běhu
        "mode": "inference",
        
        # Cesty k datům - upravte podle vašeho umístění dat
        "adc_folder": "/path/to/adc_data",
        "z_folder": "/path/to/z_adc_data",
        "label_folder": "/path/to/label_data",  # volitelné
        
        # Parametry modelu
        "model_name": "swinunetr",
        "model_path": "/path/to/trained_model.pth",
        "in_channels": 2,
        "out_channels": 2,
        
        # Parametry inference
        "inference_mode": "standard",  # nebo "moe"
        "use_tta": True,
        "training_mode": "full_volume",  # nebo "patch"
        
        # Pro MoE inferenci
        # "expert_model_path": "/path/to/expert_model.pth",
        # "moe_threshold": 80,
        
        # Výstupní adresář
        "output_dir": "outputs/example_inference",
    })
    
    # Spuštění inference
    run_inference(config)

if __name__ == "__main__":
    main() 