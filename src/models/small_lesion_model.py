import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallUNet3D(nn.Module):
    """
    Lehký 3D UNet optimalizovaný pro detekci malých lézí - může pracovat s menšími patchi.
    
    Tento model má méně vrstev a menší počet parametrů než standardní UNet,
    což umožňuje efektivní práci s menšími oblastmi a detekci drobných struktur.
    """
    def __init__(self, in_channels=2, out_channels=2, base_channels=16):
        """
        Args:
            in_channels (int): Počet vstupních kanálů
            out_channels (int): Počet výstupních kanálů (tříd)
            base_channels (int): Základní počet filtrů (výchozí: 16)
        """
        super(SmallUNet3D, self).__init__()
        
        # Počet kanálů v každé úrovni
        f = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        # Encoder
        self.enc1 = self._make_enc_block(in_channels, f[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._make_enc_block(f[0], f[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = self._make_enc_block(f[1], f[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._make_enc_block(f[2], f[3])
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = self._make_dec_block(f[3], f[2])
        
        self.upconv2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = self._make_dec_block(f[2], f[1])
        
        self.upconv1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = self._make_dec_block(f[1], f[0])
        
        # Výstupní vrstva
        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)
        
    def _make_enc_block(self, in_ch, out_ch):
        """Vytvoří enkodérový blok s jednou konvolucí a BatchNorm"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_dec_block(self, in_ch, out_ch):
        """Vytvoří dekodérový blok s jednou konvolucí a BatchNorm"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        bottleneck = self.bottleneck(p3)
        
        # Decoder s přeskakujícími spojeními
        u3 = self.upconv3(bottleneck)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        # Výstupní predikce
        out = self.out_conv(d1)
        
        return out


class SimpleResUNet3D(nn.Module):
    """
    Jednoduchý 3D UNet s residuálními spojeními, optimalizovaný pro detekci malých lézí.
    
    Tento model přidává residuální spojení pro lepší tok gradientu, což umožňuje efektivnější 
    trénování a detekci malých struktur.
    """
    def __init__(self, in_channels=2, out_channels=2, base_channels=16):
        """
        Args:
            in_channels (int): Počet vstupních kanálů
            out_channels (int): Počet výstupních kanálů (tříd)
            base_channels (int): Základní počet filtrů (výchozí: 16)
        """
        super(SimpleResUNet3D, self).__init__()
        
        # Počet kanálů v každé úrovni
        f = [base_channels, base_channels*2, base_channels*4]
        
        # Encoder
        self.enc1 = self._make_res_block(in_channels, f[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._make_res_block(f[0], f[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._make_res_block(f[1], f[2])
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = self._make_res_block(f[1]*2, f[1])
        
        self.upconv1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = self._make_res_block(f[0]*2, f[0])
        
        # Výstupní vrstva
        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)
        
    def _make_res_block(self, in_ch, out_ch):
        """Vytvoří residuální blok s konvolucemi a shortcut spojením"""
        layers = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch)
        )
        
        # Shortcut spojení pro přizpůsobení počtu kanálů
        if in_ch != out_ch:
            shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm3d(out_ch)
            )
        else:
            shortcut = nn.Identity()
        
        return lambda x: F.relu(layers(x) + shortcut(x), inplace=True)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        bottleneck = self.bottleneck(p2)
        
        # Decoder s přeskakujícími spojeními
        u2 = self.upconv2(bottleneck)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        # Výstupní predikce
        out = self.out_conv(d1)
        
        return out


def create_small_lesion_model(model_name="small_unet", in_channels=2, out_channels=2):
    """
    Vytvoří model pro detekci malých lézí podle zadaného jména.
    
    Args:
        model_name (str): Jméno modelu ("small_unet", "simple_resunet", "deeplabv3plus", "nnunet")
        in_channels (int): Počet vstupních kanálů
        out_channels (int): Počet výstupních kanálů
        
    Returns:
        nn.Module: Instance vytvořeného modelu
    """
    if model_name == "small_unet":
        return SmallUNet3D(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == "simple_resunet":
        return SimpleResUNet3D(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == "unet":
        from .unet3d import UNet3D
        return UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=16,
            bilinear=False
        )
    
    elif model_name == "deeplabv3plus":
        # Pro DeepLabV3+ je třeba doinstalovat knihovnu segmentation_models_pytorch 
        # s podporou 3D dat, což může být složitější, proto používáme zjednodušenou verzi
        raise NotImplementedError("DeepLabV3+ pro 3D data není momentálně implementován.")
    
    elif model_name == "nnunet":
        # Implementace nnUNet by vyžadovala integraci s knihovnou nnU-Net, což přesahuje rozsah této implementace
        raise NotImplementedError("nnUNet integrace není momentálně implementována.")
    
    else:
        raise ValueError(f"Neplatné jméno modelu pro detekci malých lézí: {model_name}") 