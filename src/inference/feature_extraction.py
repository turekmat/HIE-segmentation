import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Wrapper kolem SwinUNETR modelu pro extrakci příznaků z různých úrovní.
    """
    def __init__(self, model, return_layers=None):
        super(FeatureExtractor, self).__init__()
        self.model = model
        
        # Pokud nejsou specifikovány vrstvy, extrahujeme ze všech standardních míst
        if return_layers is None:
            # Pro SwinUNETR chceme extrahovat z vrstev enkodéru a bottlenecku
            self.return_layers = {'encoder1': 0, 'encoder2': 1, 'encoder3': 2, 'encoder4': 3}
        else:
            self.return_layers = return_layers
        
        # Nastavíme model do eval módu pro extrakci příznaků
        self.model.eval()
        
        # Registrujeme hooks pro zachycení příznaků
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Registruje hooks pro zachycení příznaků z SwinUNETR."""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # SwinUNETR má specifickou strukturu - musíme získat příznaky z klíčových míst
        # Tato implementace je specifická pro SwinUNETR
        if hasattr(self.model, 'swinViT'):
            # Přidáme hook pro výstupy SwinViT, které jsou použity jako skip connections
            for i, layer_name in enumerate(['encoder1', 'encoder2', 'encoder3', 'encoder4']):
                if i < len(self.model.swinViT.layers):
                    self.model.swinViT.layers[i].register_forward_hook(get_activation(layer_name))
    
    def _clear_features(self):
        """Vymaže uložené příznaky z předchozího průchodu."""
        self.features.clear()
    
    def get_features(self):
        """
        Vrátí uložené příznaky v pořadí od nejnižší po nejvyšší úroveň.
        
        Returns:
            List tensorů příznaků [level1, level2, level3, bottleneck]
        """
        # Seřadíme příznaky podle úrovně
        ordered_features = []
        for name in ['encoder1', 'encoder2', 'encoder3', 'encoder4']:
            if name in self.features:
                ordered_features.append(self.features[name])
        
        return ordered_features
    
    def forward(self, x):
        """
        Provede průchod modelem a uloží příznaky z požadovaných vrstev.
        
        Args:
            x: Vstupní tensor
            
        Returns:
            tuple: (výstup modelu, seznam příznaků)
        """
        self._clear_features()
        output = self.model(x)
        return output, self.get_features()


def extract_swinunetr_features(model, x):
    """
    Extrahuje příznaky ze SwinUNETR modelu pro feature fusion.
    
    Args:
        model: Instance SwinUNETR modelu
        x: Vstupní tensor
        
    Returns:
        tuple: (výstup modelu, seznam příznaků)
    """
    # Obalíme model naším feature extraktorem
    extractor = FeatureExtractor(model)
    
    # Nastavíme model do eval módu
    model.eval()
    
    # Extrahujeme příznaky
    with torch.no_grad():
        output, features = extractor(x)
    
    return output, features


# Funkce pro použití v inferenci
def infer_with_feature_extraction(main_model, small_model, input_vol, device="cuda"):
    """
    Provede inferenci s hlavním a menším modelem s feature fusion.
    
    Args:
        main_model: Hlavní model (SwinUNETR)
        small_model: Menší model (AttentionResUNet)
        input_vol: Vstupní tensor
        device: Zařízení pro výpočet
        
    Returns:
        dict: Slovník s predikcemi a dalšími informacemi
    """
    # Převedeme vstup na tensor
    if not isinstance(input_vol, torch.Tensor):
        input_vol = torch.tensor(input_vol, dtype=torch.float32)
    
    # Přidáme batch dimenzi, pokud chybí
    if input_vol.dim() == 4:
        input_vol = input_vol.unsqueeze(0)
    
    # Přesuneme na správné zařízení
    input_vol = input_vol.to(device)
    
    # Extrahujeme příznaky a získáme predikci hlavního modelu
    with torch.no_grad():
        main_output, main_features = extract_swinunetr_features(main_model, input_vol)
        
        # Získáme predikci malého modelu s feature fusion
        small_output = small_model(input_vol, main_features)
        
        # Převedeme výstupy zpět na CPU pro další zpracování
        main_pred = main_output.cpu().numpy()
        small_pred = small_output.cpu().numpy()
    
    return {
        "main_prediction": main_pred,
        "small_prediction": small_pred,
        "main_features": main_features
    } 