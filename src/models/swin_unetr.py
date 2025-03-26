import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

def create_model(model_name="SwinUNETR", in_channels=2, out_channels=2, drop_rate=0.15):
    """
    Vytvoří model podle zadaného jména
    
    Args:
        model_name (str): Jméno modelu (SwinUNETR, AttentionUNet, UNet3Plus, UNet3D)
        in_channels (int): Počet vstupních kanálů
        out_channels (int): Počet výstupních kanálů
        drop_rate (float): Dropout rate pro SwinUNETR
        
    Returns:
        nn.Module: Instance vytvořeného modelu
    """
    if model_name == "SwinUNETR":
        model = SwinUNETR(
            img_size=(64, 64, 64),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,
            use_checkpoint=False,
            spatial_dims=3,
            drop_rate=drop_rate,
        )
    elif model_name == "AttentionUNet":
        from .attention_unet import AttentionUNet
        model = AttentionUNet(
          in_channels=in_channels,
          out_channels=out_channels)

    elif model_name == "UNet3Plus":
        from .unet3plus import UNet3Plus3D
        model = UNet3Plus3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64
        )
    elif model_name == "UNet3D":
        from .unet3d import UNet3D
        model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
            bilinear=False
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model 