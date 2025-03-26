import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Dvojitá konvoluce: 3D konvoluce -> BatchNorm -> ReLU -> 3D konvoluce -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling a dvojitá konvoluce
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling a dvojitá konvoluce
    """
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=False):
        """
        Args:
            in_channels: Počet kanálů prvního vstupu (z bottlenecku)
            mid_channels: Počet kanálů druhého vstupu (ze skip connection)
            out_channels: Počet výstupních kanálů
            bilinear: Použít bilineární interpolaci místo transponované konvoluce
        """
        super(Up, self).__init__()

        # Pokud bilinear=True, použije se bilineární interpolace místo transponované konvoluce
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            # Transponovaná konvoluce pro upsampling
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2)

        # Po spojení bottleneck featury (in_channels//2) a skip featury (mid_channels)
        self.conv = DoubleConv(in_channels // 2 + mid_channels, out_channels)

    def forward(self, x1, x2):
        # x1 je z nižší úrovně (bottleneck)
        # x2 je ze skip connection (vyšší úroveň)
        x1 = self.up(x1)
        
        # Výpočet paddingu pro zachování správných rozměrů
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
            diffZ // 2, diffZ - diffZ // 2
        ])
        
        # Spojení podél kanálové osy
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Výstupní konvoluce
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Klasický 3D UNet pro segmentaci medicínských obrazů.
    """
    def __init__(self, in_channels=2, out_channels=2, base_channels=32, bilinear=False):
        """
        Args:
            in_channels (int): Počet vstupních kanálů
            out_channels (int): Počet výstupních kanálů (tříd)
            base_channels (int): Základní počet filtrů (výchozí: 32)
            bilinear (bool): Použít bilineární interpolaci místo transponované konvoluce
        """
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Základní počet kanálů a faktoru downsamplingu
        f = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        factor = 2 if bilinear else 1
        
        # Encodér
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        
        # Bottleneck
        if bilinear:
            self.down4 = Down(f[3], f[4] // factor)
            bottleneck_channels = f[4] // factor
        else:
            self.down4 = Down(f[3], f[4])
            bottleneck_channels = f[4]
        
        # Dekodér - upraveny počty kanálů pro správnou inicializaci
        self.up1 = Up(bottleneck_channels, f[3], f[3], bilinear)
        self.up2 = Up(f[3], f[2], f[2], bilinear)
        self.up3 = Up(f[2], f[1], f[1], bilinear)
        self.up4 = Up(f[1], f[0], f[0], bilinear)
        
        # Výstupní vrstva
        self.outc = OutConv(f[0], out_channels)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder + skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Výstupní konvoluce
        logits = self.outc(x)
        return logits 