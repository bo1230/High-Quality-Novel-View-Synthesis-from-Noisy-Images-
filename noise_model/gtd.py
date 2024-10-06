import sys

sys.path.append('../')

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
import numpy as np


def make_model(args):
    return GDTD(args), 1


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(3):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))    
            if bn:
                m.append(nn.BatchNorm2d(n_feats))   
            if i%2 == 0:
                m.append(act)   

        self.body = nn.Sequential(*m)     
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)  
        res += x                                

        return res


class S_ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(S_ResBlock, self).__init__()

        assert len(conv) == 2

        m = []

        for i in range(2):
            m.append(conv[i](n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,  
                 num_classes: int = 3,  
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits

class GDTD(nn.Module):

    def __init__(self, H, W, patch_dim, conv=default_conv):
        super(GDTD, self).__init__()

        self.scale_idx = 0
        n_feats = 48
        n_colors = 3
        kernel_size = 3
        act = nn.ReLU(True)

        self.head1 = conv(n_colors, n_feats, kernel_size)
        self.noise_extract = UNet(3,3)
        self.head1_1 = ResBlock(conv, 64, kernel_size, act=act)

        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, 1, bias=False)
        )

    def get_2d_emb(self, batch_size, x, y, out_ch, device): 
        out_ch = int(np.ceil(out_ch / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch)).cuda()  
        pos_x = torch.arange(x, device=device).type(inv_freq.type()) * 2 * np.pi / x
        pos_y = torch.arange(y, device=device).type(inv_freq.type()) * 2 * np.pi / y
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq).cuda() 
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq).cuda()
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).cuda()  
        emb_y = self.get_emb(sin_inp_y).cuda()
        emb = torch.zeros((x, y, out_ch * 2), device=device) 
        emb[:, :, : out_ch] = emb_x
        emb[:, :, out_ch: 2 * out_ch] = emb_y
        return emb 

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, args, image, viewpoint_cam):

        shuffle_rgb = image.unsqueeze(0)
        pos_enc = self.get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, device="cuda")
     
        noise1 = self.noise_extract(viewpoint_cam.original_image.unsqueeze(0))  

        x = self.head1(image)
        x = torch.cat([x, pos_enc.permute(2, 0, 1)], 0)  # 48+16=64
        x = self.head1_1(x)    
        mask = self.mlp(x)    
        mask = torch.sigmoid(mask)

        noise = mask * (viewpoint_cam.pre_noise) + (1 - mask) * noise1.squeeze()
        noise_img = image + noise
        return noise_img, noise, mask

