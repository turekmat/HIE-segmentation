import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3Plus3D(nn.Module):
    """
    A 3D adaptation of UNet 3+ for volume segmentation.
    This is a simplified version focusing on the full-scale skip connection
    to the highest (original) resolution only (i.e. 'D0' aggregator).
    """
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=64
    ):
        """
        Args:
            in_channels  (int): number of input channels (e.g., 2 => [ADC, Z_ADC])
            out_channels (int): number of output channels (e.g., 2 => [BG, FG])
            base_channels(int): number of channels in the first UNet level.
                                (subsequent levels use multiples of this)
        """
        super(UNet3Plus3D, self).__init__()

        filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16
        ]  # [64, 128, 256, 512, 1024] by default

        # --- Encoder ---
        self.enc0 = self._make_enc_block(in_channels, filters[0])   # level 0
        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = self._make_enc_block(filters[0], filters[1])    # level 1
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = self._make_enc_block(filters[1], filters[2])    # level 2
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = self._make_enc_block(filters[2], filters[3])    # level 3
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = self._make_enc_block(filters[3], filters[4])    # level 4 (deepest)

        self.conv0_0 = nn.Conv3d(filters[0], base_channels, kernel_size=3, padding=1)
        self.conv1_0 = nn.Conv3d(filters[1], base_channels, kernel_size=3, padding=1)
        self.conv2_0 = nn.Conv3d(filters[2], base_channels, kernel_size=3, padding=1)
        self.conv3_0 = nn.Conv3d(filters[3], base_channels, kernel_size=3, padding=1)
        self.conv4_0 = nn.Conv3d(filters[4], base_channels, kernel_size=3, padding=1)

        # BatchNorm + activation for aggregator outputs
        self.bn0_0 = nn.BatchNorm3d(base_channels)
        self.bn1_0 = nn.BatchNorm3d(base_channels)
        self.bn2_0 = nn.BatchNorm3d(base_channels)
        self.bn3_0 = nn.BatchNorm3d(base_channels)
        self.bn4_0 = nn.BatchNorm3d(base_channels)

        # Final conv after concatenation => combine 5*base_channels => 320 if base_channels=64
        self.agg_conv = nn.Sequential(
            nn.Conv3d(base_channels * 5, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True)
        )

        # Final classification layer => out_channels
        self.out_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def _make_enc_block(self, in_ch, out_ch):
        """
        Basic 3D U-Net encoder block: (Conv3D+BN+ReLU) x 2
        """
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x shape: [B, in_channels, D, H, W]
        """
        # --- Encoder downsampling ---
        x0 = self.enc0(x)             # -> (B, filters[0], D, H, W)
        x1 = self.enc1(self.pool0(x0))# -> (B, filters[1], D/2, H/2, W/2)
        x2 = self.enc2(self.pool1(x1))# -> (B, filters[2], D/4, H/4, W/4)
        x3 = self.enc3(self.pool2(x2))# -> (B, filters[3], D/8, H/8, W/8)
        x4 = self.enc4(self.pool3(x3))# -> (B, filters[4], D/16,H/16,W/16)

        # a) from x0 (same size, no upsample)
        hx0 = F.relu(self.bn0_0(self.conv0_0(x0)), inplace=True)

        # b) from x1 (2x upsample)
        hx1_ = F.interpolate(x1, scale_factor=2, mode='trilinear', align_corners=True)
        hx1 = F.relu(self.bn1_0(self.conv1_0(hx1_)), inplace=True)

        # c) from x2 (4x upsample)
        hx2_ = F.interpolate(x2, scale_factor=4, mode='trilinear', align_corners=True)
        hx2 = F.relu(self.bn2_0(self.conv2_0(hx2_)), inplace=True)

        # d) from x3 (8x upsample)
        hx3_ = F.interpolate(x3, scale_factor=8, mode='trilinear', align_corners=True)
        hx3 = F.relu(self.bn3_0(self.conv3_0(hx3_)), inplace=True)

        # e) from x4 (16x upsample)
        hx4_ = F.interpolate(x4, scale_factor=16, mode='trilinear', align_corners=True)
        hx4 = F.relu(self.bn4_0(self.conv4_0(hx4_)), inplace=True)

        # Concat
        d0 = torch.cat([hx0, hx1, hx2, hx3, hx4], dim=1)
        d0 = self.agg_conv(d0)        # -> (B, filters[0], D, H, W)

        # Final output segmentation
        out = self.out_conv(d0)       # -> (B, out_channels, D, H, W)

        return out 