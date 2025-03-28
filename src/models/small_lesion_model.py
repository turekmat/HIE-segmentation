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


class ResBlock(nn.Module):
    """
    Residuální blok, který řeší problém s lambda funkcemi při přenosu na GPU
    """
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch)
        )
        
        # Shortcut spojení pro přizpůsobení počtu kanálů
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return F.relu(self.layers(x) + self.shortcut(x), inplace=True)


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
        self.enc1 = ResBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ResBlock(f[0], f[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ResBlock(f[1], f[2])
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = ResBlock(f[1]*2, f[1])
        
        self.upconv1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = ResBlock(f[0]*2, f[0])
        
        # Výstupní vrstva
        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)
    
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


class AttentionGate(nn.Module):
    """
    Attention Gate pro 3D Attention U-Net.
    Implementace založena na článku: 
    "Attention U-Net: Learning Where to Look for the Pancreas"
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g = features z dekodéru, x = skip connection features
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Zajištění stejné velikosti tenzorů před sčítáním
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Zajištění stejné velikosti tenzoru psi s tenzorem x před násobením
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='trilinear', align_corners=True)
            
        return x * psi


class SmallAttentionUNet3D(nn.Module):
    """
    3D Attention U-Net optimalizovaný pro malé patche a detekci malých lézí.
    
    Tento model využívá attention mechanismus k zaměření na relevantní části vstupních dat,
    což je zvláště užitečné pro detekci malých lézí v 3D objemech.
    
    Optimalizovaný pro patch size 16x16x16 s menším počtem úrovní downsamplingu.
    """
    def __init__(self, in_channels=2, out_channels=2, base_channels=16):
        """
        Args:
            in_channels (int): Počet vstupních kanálů
            out_channels (int): Počet výstupních kanálů (tříd)
            base_channels (int): Základní počet filtrů (výchozí: 16)
        """
        super(SmallAttentionUNet3D, self).__init__()
        
        # Počet kanálů v každé úrovni - pouze 3 úrovně pro malé patche
        f = [base_channels, base_channels*2, base_channels*4]
        
        # Encoder
        self.enc1 = self._make_enc_block(in_channels, f[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._make_enc_block(f[0], f[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck - na této úrovni bude feature mapa 4x4x4 pro patch 16x16x16
        self.bottleneck = self._make_enc_block(f[1], f[2])
        
        # Attention gates
        self.att1 = AttentionGate(F_g=f[1], F_l=f[0], F_int=f[0]//2)
        self.att2 = AttentionGate(F_g=f[2], F_l=f[1], F_int=f[1]//2)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = self._make_dec_block(f[1]*2, f[1])
        
        self.upconv1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = self._make_dec_block(f[0]*2, f[0])
        
        # Výstupní vrstva
        self.out_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)
        
        # Dropout pro regularizaci
        self.dropout = nn.Dropout3d(0.3)
        
    def _make_enc_block(self, in_ch, out_ch):
        """Vytvoří enkodérový blok s dvojitou konvolucí a BatchNorm"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_dec_block(self, in_ch, out_ch):
        """Vytvoří dekodérový blok s dvojitou konvolucí a BatchNorm"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        e2 = self.dropout(e2)  # Dropout pro lepší generalizaci
        p2 = self.pool2(e2)
        
        # Bottleneck
        bottleneck = self.bottleneck(p2)
        bottleneck = self.dropout(bottleneck)  # Dropout pro lepší generalizaci
        
        # Decoder s attention gates a přeskakujícími spojeními
        # Attention mezi bottleneck a encoder 2
        att2 = self.att2(bottleneck, e2)
        
        u2 = self.upconv2(bottleneck)
        u2 = torch.cat([u2, att2], dim=1)
        d2 = self.dec2(u2)
        
        # Attention mezi decoder 2 a encoder 1
        att1 = self.att1(d2, e1)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, att1], dim=1)
        d1 = self.dec1(u1)
        
        # Výstupní predikce
        out = self.out_conv(d1)
        
        return out


def create_small_lesion_model(model_name="small_unet", in_channels=2, out_channels=2):
    """
    Vytvoří model pro detekci malých lézí podle zadaného jména.
    
    Args:
        model_name (str): Jméno modelu ("small_unet", "simple_resunet", "attention_unet", "unet")
        in_channels (int): Počet vstupních kanálů
        out_channels (int): Počet výstupních kanálů
        
    Returns:
        nn.Module: Instance vytvořeného modelu
    """
    if model_name == "small_unet":
        return SmallUNet3D(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == "simple_resunet":
        return SimpleResUNet3D(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == "attention_unet":
        return SmallAttentionUNet3D(in_channels=in_channels, out_channels=out_channels, base_channels=24)
    
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