from .metrics import compute_masd, compute_nsd

def setup_wandb(project_name, run_name, config):
    """
    Inicializuje Weights & Biases pro logování experimentů.
    
    Args:
        project_name (str): Název projektu ve Weights & Biases
        run_name (str): Název běhu (runu) ve Weights & Biases
        config (dict): Konfigurační slovník s parametry experimentu
        
    Returns:
        wandb.run: Objekt běhu Weights & Biases
    """
    import wandb
    try:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )
        print(f"Weights & Biases inicializován pro projekt '{project_name}', běh '{run_name}'")
        return run
    except Exception as e:
        print(f"Chyba při inicializaci Weights & Biases: {e}")
        print("Wandb logování bude vypnuto.")
        return None
