import torch
import torch.nn as nn
import torch.nn.functional as F

def create_feature_maps(init_channel_number, number_of_fmaps):
    """
    Vygeneruje list velikostí feature map.
    Defaultně pro 6 úrovní (můžete přizpůsobit).
    """
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class SCA3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCA3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, d, h, w = x.size()
        chn_se = self.avg_pool(x).view(b, c)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se))
        chn_se = chn_se.view(b, c, 1, 1, 1)
        x_channel_att = x * chn_se

        spa_se = torch.sigmoid(self.spatial_se(x))
        x_spatial_att = x * spa_se

        out = x + x_channel_att + x_spatial_att
        return out


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    assert 'c' in order, "Konvoluce (c) musí být ve stringu order"
    assert order[0] not in 'rle', "Nejprve musí být konvoluce, pak až aktivace"

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Neznámý znak vrstvy '{char}'. Povolené: ['b','g','r','l','e','c']")

    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super().__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        encoder=True,
        kernel_size=3,
        order='crg',
        num_groups=8
    ):
        super().__init__()
        if encoder:
            # Zpravidla pro enkodér
            conv1_out = max(in_channels, out_channels // 2)
            self.add_module('SingleConv1', SingleConv(in_channels, conv1_out, kernel_size, order, num_groups))
            self.add_module('SingleConv2', SingleConv(conv1_out, out_channels, kernel_size, order, num_groups))
        else:
            # Dekodér
            self.add_module('SingleConv1', SingleConv(in_channels, out_channels, kernel_size, order, num_groups))
            self.add_module('SingleConv2', SingleConv(out_channels, out_channels, kernel_size, order, num_groups))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=(2, 2, 2),
        pool_type='max',
        basic_module=DoubleConv,
        conv_layer_order='crg',
        num_groups=8
    ):
        super().__init__()
        assert pool_type in ['max', 'avg'], "pool_type musí být 'max' nebo 'avg'"

        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        scale_factor=(2, 2, 2),
        basic_module=DoubleConv,
        conv_layer_order='crg',
        num_groups=8
    ):
        super().__init__()
        self.upsample = None  # v originále by šlo nastavit ConvTranspose...

        self.scse = SCA3D(in_channels)

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=kernel_size,
            order=conv_layer_order,
            num_groups=num_groups
        )

    def forward(self, encoder_features, x):
        if self.upsample is None:
            # nearest neighbor upsampling
            out_size = encoder_features.size()[2:]  # D,H,W
            x = F.interpolate(x, size=out_size, mode='nearest')
            # skip: spojit enkodérové a dekód. feature mapy
            x = torch.cat((encoder_features, x), dim=1)
        else:
            # pokud byste chtěli ConvTranspose3d => x = self.upsample(x) ; x += encoder_features
            raise NotImplementedError("V této verzi se pro upsampling používá nearest+concat.")

        x = self.scse(x)
        x = self.basic_module(x)
        return x


class AttentionUNet(nn.Module):
    """
    3D U-Net s "attention" (SCA3D) v dekodéru.
    - in_channels: počet vstupních kanálů (např. 2 => [ADC, Z_ADC])
    - out_channels: počet výstupních kanálů (pro segmentaci => 2 => [BG, FG])
    - final_sigmoid: zda aplikovat Sigmoid na výstup (True pro binární, False -> Softmax)
    - f_maps: základní počet feature map (16, 32, 64, 128, 256, 512 atp.)
    - layer_order: např. "crg" => Conv + ReLU + GroupNorm
    - num_groups: pro GroupNorm
    """
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        final_sigmoid=False,
        f_maps=16,
        layer_order='crg',
        num_groups=8,
        **kwargs
    ):
        super().__init__()

        if isinstance(f_maps, int):
            # např. 16, 32, 64, 128, 256, 512 atd. (6 úrovní)
            f_maps = create_feature_maps(f_maps, number_of_fmaps=6)

        # --- ENCODERS ---
        encoders = []
        for i, out_feat_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels, out_feat_num,
                    apply_pooling=False,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    num_groups=num_groups
                )
            else:
                encoder = Encoder(
                    f_maps[i - 1], out_feat_num,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    num_groups=num_groups
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f = list(reversed(f_maps))
        for i in range(len(reversed_f) - 1):
            in_chan = reversed_f[i] + reversed_f[i + 1]
            out_chan= reversed_f[i + 1]
            decoder = Decoder(
                in_channels=in_chan,
                out_channels=out_chan,
                basic_module=DoubleConv,
                conv_layer_order=layer_order,
                num_groups=num_groups
            )
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x shape: [B, in_channels, D, H, W]
        """
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.insert(0, x)

        bottom = encoder_features[0]
        skip_list = encoder_features[1:]

        x = bottom
        for decoder, skip_f in zip(self.decoders, skip_list):
            x = decoder(skip_f, x)

        x = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)

        return x 