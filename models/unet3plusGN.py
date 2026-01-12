import torch
import torch.nn as nn

class UNet_3Plus_GN(nn.Module):
    def __init__(self,
                 **kwargs):
        super(UNet_3Plus_GN, self).__init__()
        self.model = UNet_3Plus_GN(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)

def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """
    Pick the largest group count <= max_groups that divides num_channels.
    Falls back to 1 (LayerNorm-like over channels) if nothing divides.
    """
    max_groups = min(max_groups, num_channels)
    for g in range(max_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1

def GN(num_channels: int, max_groups: int = 32, eps: float = 1e-5, affine: bool = True) -> nn.GroupNorm:
    return nn.GroupNorm(_choose_gn_groups(num_channels, max_groups), num_channels, eps=eps, affine=affine)

class UNetConv2_GN(nn.Module):
    """
    A simple UNet-style double conv block with GroupNorm.
    (Conv -> GN -> ReLU) x 2
    """
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            GN(out_ch, max_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            GN(out_ch, max_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

def init_weights(m: nn.Module, init_type: str = "kaiming") -> None:
    """
    Minimal init helper (compatible with the original code style).
    - Conv2d: Kaiming normal
    - GroupNorm: weight=1, bias=0
    """
    if isinstance(m, nn.Conv2d):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class UNet_3Plus_GN(nn.Module):
    """
    UNet 3+ (no deep supervision) with GroupNorm instead of BatchNorm.

    Notes:
      - Keeps the same forward topology as typical UNet3+ official code.
      - `final_activation=None` returns logits (recommended with BCEWithLogitsLoss).
      - `final_activation="sigmoid"` returns sigmoid output (official-style).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        feature_scale: int = 4,   # kept for API-compatibility (not used in this GN port)
        is_deconv: bool = True,
        is_batchnorm: bool = False,  # kept for API-compatibility (GN version ignores this)
        final_activation: str | None = "sigmoid",  # None or "sigmoid"
        gn_groups: int = 32,
    ):
        super().__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.final_activation = final_activation
        self.gn_groups = gn_groups

        filters = [64, 128, 256, 512, 1024]

        # -------------Encoder--------------
        self.conv1 = UNetConv2_GN(self.in_channels, filters[0], gn_groups=self.gn_groups)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2_GN(filters[0], filters[1], gn_groups=self.gn_groups)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2_GN(filters[1], filters[2], gn_groups=self.gn_groups)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2_GN(filters[2], filters[3], gn_groups=self.gn_groups)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = UNetConv2_GN(filters[3], filters[4], gn_groups=self.gn_groups)

        # -------------Decoder--------------
        self.CatChannels = filters[0]   # 64
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks  # 320

        # stage 4d
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, bias=False)
        self.h1_PT_hd4_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, bias=False)
        self.h2_PT_hd4_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, bias=False)
        self.h3_PT_hd4_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1, bias=False)
        self.h4_Cat_hd4_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd4_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.gn4d_1 = GN(self.UpChannels, max_groups=self.gn_groups)
        self.relu4d_1 = nn.ReLU(inplace=True)

        # stage 3d
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, bias=False)
        self.h1_PT_hd3_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, bias=False)
        self.h2_PT_hd3_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, bias=False)
        self.h3_Cat_hd3_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd4_UT_hd3_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd3_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.gn3d_1 = GN(self.UpChannels, max_groups=self.gn_groups)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # stage 2d
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, bias=False)
        self.h1_PT_hd2_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, bias=False)
        self.h2_Cat_hd2_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd3_UT_hd2_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd4_UT_hd2_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd2_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.gn2d_1 = GN(self.UpChannels, max_groups=self.gn_groups)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # stage 1d
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, bias=False)
        self.h1_Cat_hd1_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd2_UT_hd1_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd3_UT_hd1_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, bias=False)
        self.hd4_UT_hd1_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, bias=False)
        self.hd5_UT_hd1_gn = GN(self.CatChannels, max_groups=self.gn_groups)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=False)
        self.gn1d_1 = GN(self.UpChannels, max_groups=self.gn_groups)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights (official-style-ish)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.GroupNorm)):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # -------------Encoder-------------
        h1 = self.conv1(inputs)
        h2 = self.conv2(self.maxpool1(h1))
        h3 = self.conv3(self.maxpool2(h2))
        h4 = self.conv4(self.maxpool3(h3))
        hd5 = self.conv5(self.maxpool4(h4))

        # -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_gn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_gn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_gn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_gn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_gn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.gn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_gn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_gn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_gn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_gn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_gn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.gn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_gn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_gn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_gn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_gn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_gn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.gn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_gn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_gn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_gn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_gn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_gn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.gn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))

        d1 = self.outconv1(hd1)

        if self.final_activation is None:
            return d1
        if isinstance(self.final_activation, str) and self.final_activation.lower() == "sigmoid":
            return torch.sigmoid(d1)
        raise ValueError(f"Unsupported final_activation: {self.final_activation}")

if __name__ == "__main__":
    # quick sanity check
    m = UNet_3Plus_GN(in_channels=3, n_classes=1, final_activation=None)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print("Output:", y.shape)
