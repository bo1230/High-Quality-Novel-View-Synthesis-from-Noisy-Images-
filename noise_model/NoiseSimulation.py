import numpy as np
import os
# from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn


class NoiseSimulation(nn.Module):
    def __init__(self, num_img, H, W, img_embed=32):
        super().__init__()
        self.num_img = num_img  # 图像编号
        self.W, self.H = W, H  # 图像宽高

        self.img_embed_cnl = img_embed  # 图像嵌入通道

        # self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4  # 频率相关

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl)  # 图像嵌入层，使用 Embedding 层

        self.mlp_base = torch.nn.Sequential(
            torch.nn.Conv2d(48+32, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
        )  # 基础mlp网络


        # 学习乘性噪声和加性噪声
        hiddens = [torch.nn.Linear(64, 64) if i % 2 == 0 else torch.nn.ReLU()
                   for i in range((64 - 1) * 2)]
        self.noiseParparameter = torch.nn.Sequential(
                torch.nn.Linear(H * W * 3, 64), torch.nn.ReLU(),
                *hiddens,
                torch.nn.Linear(64, 2),
            )



        # self.mlp_head1 = torch.nn.Conv2d(64, ks ** 2, 1, bias=False)  # 头部卷积层
        self.coarse_noise = torch.nn.Conv2d(64, 3, 1, bias=False)  # 掩码卷积层
        # self.noise_weight = torch.nn.Conv2d(64, 3, 1, bias=False)  # 掩码卷积层
        #
        self.conv_rgb = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(64, 32, 3, padding=1)
        )  # RGBD 特征提取卷积层
    # def noiseModel(self, rgb, sig_read, sig_shot):
    #
    #     std = (sig_read ** 2 + sig_shot ** 2 * rgb) ** 0.5
    #     noise = std * torch.randn_like(rgb)     # 生成与rgb形状相同的随机噪声（服从正态分布），
    #     noise_rgb = rgb + noise
    #
    #     return noise_rgb

    def forward(self, viewpoint_cam, img_idx, pos_enc, img, gt_image):
        img_noise = img + viewpoint_cam.noise
        rgb_noise_feat = self.conv_rgb(img_noise).unsqueeze(0) # 提取图像特征

        # 图像嵌入和位置编码
        img_embed = self.embedding_camera(torch.LongTensor([img_idx]).cuda())[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0], pos_enc.shape[1], pos_enc.shape[2], img_embed.shape[-1])
        inp = torch.cat([img_embed, pos_enc],-1).permute(0, 3, 1, 2)

        feat = self.mlp_base(torch.cat([inp, rgb_noise_feat],1))  # 从视角嵌入，位置嵌入和RGB特征中提取特征

        noise_refine = self.coarse_noise(feat)  # 生成模糊核
        # weight = self.noise_weight(feat)      # 生成权重
        # weight = torch.sigmoid(weight)
        # noise_refine = noise_refine * weight


        return noise_refine  # 返回权重和掩码



