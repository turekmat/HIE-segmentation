import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Implementace kanálové pozornosti pro CBAM (Convolutional Block Attention Module).
    Kanálová pozornost se zaměřuje na "co" je důležité v příznacích.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Sdílené MLP pro zpracování obou poolingových výstupů
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Implementace prostorové pozornosti pro CBAM.
    Prostorová pozornost se zaměřuje na "kde" jsou důležité příznaky.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Velikost jádra musí být 3 nebo 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Vytvoříme mapy příznaků pomocí průměru a maxima napříč kanály
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Kombinuje kanálovou a prostorovou pozornost pro lepší zaměření na relevantní části vstupu.
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)  # Aplikace kanálové pozornosti
        x = x * self.spatial_att(x)  # Aplikace prostorové pozornosti
        return x


class ResBlock(nn.Module):
    """
    Reziduální blok s CBAM.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # CBAM attention module
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_cbam:
            out = self.cbam(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    """
    Blok enkodéru pro AttentionResUNet.
    """
    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(EncoderBlock, self).__init__()
        self.res_block = ResBlock(in_channels, out_channels, use_cbam=use_cbam)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        features = self.res_block(x)
        pooled = self.pool(features)
        return pooled, features


class FeatureFusion(nn.Module):
    """
    Blok pro fúzi příznaků z hlavního modelu (SwinUNETR) s příznaky menšího modelu.
    """
    def __init__(self, small_channels, main_channels):
        super(FeatureFusion, self).__init__()
        
        # Přizpůsobení počtu kanálů z hlavního modelu
        self.adapt_conv = nn.Conv3d(main_channels, small_channels, kernel_size=1, bias=False)
        
        # Attention mechanism pro určení důležitosti příznaků z obou modelů
        self.attention = nn.Sequential(
            nn.Conv3d(small_channels * 2, small_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(small_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(small_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, small_features, main_features=None):
        # Pokud nejsou k dispozici příznaky z hlavního modelu, vrátit pouze malé příznaky
        if main_features is None:
            return small_features
        
        # Přizpůsobení dimenzí hlavních příznaků
        if main_features.shape[2:] != small_features.shape[2:]:
            main_features = F.interpolate(main_features, size=small_features.shape[2:],
                                          mode='trilinear', align_corners=True)
        
        # Přizpůsobení počtu kanálů
        main_features_adapted = self.adapt_conv(main_features)
        
        # Konkatenace příznaků pro attention mechanismus
        combined = torch.cat([small_features, main_features_adapted], dim=1)
        
        # Výpočet attention vah
        weights = self.attention(combined)
        
        # Aplikace attention vah
        small_weights = weights[:, 0:1, :, :, :]
        main_weights = weights[:, 1:2, :, :, :]
        
        fused_features = small_weights * small_features + main_weights * main_features_adapted
        
        return fused_features


class DecoderBlock(nn.Module):
    """
    Blok dekodéru pro AttentionResUNet s možností fúze příznaků.
    """
    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.fusion = FeatureFusion(in_channels // 2, in_channels // 2)
        self.res_block = ResBlock(in_channels, out_channels, use_cbam=use_cbam)
        
    def forward(self, x, skip_connection, main_features=None):
        x = self.up(x)
        
        # Zajištění shodné velikosti pro konkatenaci
        if x.shape[2:] != skip_connection.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:], 
                             mode='trilinear', align_corners=True)
        
        # Aplikace feature fusion na skip connection, pokud jsou k dispozici hlavní příznaky
        if main_features is not None:
            skip_connection = self.fusion(skip_connection, main_features)
        
        # Konkatenace s odpovídajícím výstupem z enkodéru
        x = torch.cat([x, skip_connection], dim=1)
        x = self.res_block(x)
        
        return x


class AttentionResUNet(nn.Module):
    """
    3D Attention-Augmented ResUNet s CBAM bloky a feature fusion.
    
    Tato architektura je speciálně navržena pro segmentaci malých lézí HIE
    a může být použita jako sekundární model v kaskádovém přístupu spolu s SwinUNETR.
    """
    def __init__(self, in_channels=2, out_channels=2, features=[32, 64, 128, 256],
                 use_cbam=True, enable_fusion=True):
        super(AttentionResUNet, self).__init__()
        
        self.enable_fusion = enable_fusion
        
        # Počáteční konvoluce
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Enkodér
        self.encoder1 = EncoderBlock(features[0], features[0], use_cbam=use_cbam)
        self.encoder2 = EncoderBlock(features[0], features[1], use_cbam=use_cbam)
        self.encoder3 = EncoderBlock(features[1], features[2], use_cbam=use_cbam)
        
        # Bottleneck
        self.bottleneck = ResBlock(features[2], features[3], use_cbam=use_cbam)
        
        # Dekodér
        self.decoder1 = DecoderBlock(features[3], features[2], use_cbam=use_cbam)
        self.decoder2 = DecoderBlock(features[2], features[1], use_cbam=use_cbam)
        self.decoder3 = DecoderBlock(features[1], features[0], use_cbam=use_cbam)
        
        # Konečná konvoluce
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x, main_features=None):
        """
        Forward pass s volitelnou feature fusion.
        
        Args:
            x: Vstupní tensor
            main_features: Seznam příznaků z hlavního modelu (SwinUNETR) pro každou úroveň,
                           ve formátu [level1, level2, level3, bottleneck], může být None
        """
        # Standardní průchod menším modelem
        x1 = self.initial_conv(x)
        
        # Enkodér
        x2, skip1 = self.encoder1(x1)
        x3, skip2 = self.encoder2(x2)
        x4, skip3 = self.encoder3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Dekodér s feature fusion
        if self.enable_fusion and main_features is not None:
            # Získáme příznaky z hlavního modelu pro každou úroveň
            # Pokud je délka main_features menší než potřebujeme, použijeme None
            bottleneck_feat = main_features[3] if len(main_features) > 3 else None
            level3_feat = main_features[2] if len(main_features) > 2 else None
            level2_feat = main_features[1] if len(main_features) > 1 else None
            level1_feat = main_features[0] if len(main_features) > 0 else None
            
            # Aplikace feature fusion na každé úrovni
            x = self.decoder1(x5, skip3, level3_feat)
            x = self.decoder2(x, skip2, level2_feat)
            x = self.decoder3(x, skip1, level1_feat)
        else:
            # Standardní průchod bez fusion
            x = self.decoder1(x5, skip3)
            x = self.decoder2(x, skip2)
            x = self.decoder3(x, skip1)
        
        # Konečná konvoluce
        x = self.final_conv(x)
        
        return x


def create_attention_resunet(in_channels=2, out_channels=2, features=[32, 64, 128, 256], 
                            use_cbam=True, enable_fusion=True):
    """
    Tovární funkce pro vytvoření Attention-Augmented ResUNet.
    
    Args:
        in_channels: Počet vstupních kanálů (např. 2 pro ADC a Z-ADC)
        out_channels: Počet výstupních kanálů (např. 2 pro binární segmentaci s pozadím)
        features: Seznam velikostí příznaků pro každou úroveň architektury
        use_cbam: Zda používat CBAM bloky
        enable_fusion: Zda povolit feature fusion s hlavním modelem
        
    Returns:
        Instanci modelu AttentionResUNet
    """
    return AttentionResUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        use_cbam=use_cbam,
        enable_fusion=enable_fusion
    ) 