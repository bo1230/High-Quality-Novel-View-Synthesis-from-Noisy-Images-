'''
    Group + Dynamic + transformer  + Denosing
'''

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
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        # 构建卷积层和可选的批量归一化层及激活层
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))    # 添加卷积层
            if bn:
                m.append(nn.BatchNorm2d(n_feats))   # 如果需要批量归一化，则添加批量归一化层
            if i == 0:
                m.append(act)   # 第一个卷积层后添加激活层

        self.body = nn.Sequential(*m)     # 卷积+激活+卷积
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)  # 计算残差，并按比例缩放
        res += x                                # 将残差加到输入上，实现残差连接

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


# 定义双卷积类
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 定义padding为1，保持卷积后特征图的宽度和高度不变，具体计算N= (W-F+2P)/S+1
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 加入BN层，提升训练速度，并提高模型效果
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积层，同第一个
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # BN层
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# 调用上面定义的双卷积类，定义下采样类
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            # 最大池化，步长为2，池化核大小为2，计算公式同卷积，则 N = (W-F+2P)/S+1,  N= (W-2+0)/4 + 1
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        # 调用转置卷积的方法进行上采样，使特征图的高和宽翻倍，out  =(W−1)×S−2×P+F，通道数减半
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 调用双层卷积类，通道数是否减半要看out_channels接收的值
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # X的shape为[N, C, H, W]，下面三行代码主要是为了保证x1和x2在维度为2和3的地方保持一致，方便cat操作不出错。
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 增加padding操作，padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 定义输出卷积类
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,  # 默认输入图像的通道数为1，这里一般黑白图像为1，而彩色图像为3
                 num_classes: int = 3,  # 默认输出的分类类别数为2
                 # 默认基础通道为64，这里也可以改成大于2的任意2的次幂，不过越大模型的复杂度越高，参数越大，模型的拟合能力也就越强
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # 编码器的第1个双卷积层，不包含下采样过程，输入通道为1，输出通道数为base_c,这个值可以为64或者32
        self.in_conv = DoubleConv(in_channels, base_c)
        # 编码器的第2个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down1 = Down(base_c, base_c * 2)
        # 编码器的第3个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down2 = Down(base_c * 2, base_c * 4)
        # 编码器的第4个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down3 = Down(base_c * 4, base_c * 8)
        # 编码器的第5个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down4 = Down(base_c * 8, base_c * 16)

        # 解码器的第1个上采样模块，首先进行一个转置卷积，使特征图的高和宽翻倍，通道数减半；
        # 对x1（x1可以到总的forward函数中可以知道它代指什么）进行padding，使其与x2的尺寸一致，然后在第1维通道维度进行concat，通道数翻倍。
        # 最后再进行一个双卷积层，通道数减半，高和宽不变。
        self.up1 = Up(base_c * 16, base_c * 8)
        # 解码器的第2个上采样模块，操作同上
        self.up2 = Up(base_c * 8, base_c * 4)
        # 解码器的第3个上采样模块，操作同上
        self.up3 = Up(base_c * 4, base_c * 2)
        # 解码器的第4个上采样模块，操作同上
        self.up4 = Up(base_c * 2, base_c)
        # 解码器的输出卷积模块，改变输出的通道数为分类的类别数
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        # 假设输入的特征图尺寸为[N, C, H, W]，[4, 3, 480, 480],依次代表BatchSize, 通道数量，高，宽；   则输出为[4, 64, 480,480]
        x1 = self.in_conv(x)
        # 输入的特征图尺寸为[4, 64, 480, 480];  输出为[4, 128, 240,240]
        x2 = self.down1(x1)
        # 输入的特征图尺寸为[4, 128, 240,240];  输出为[4, 256, 120,120]
        x3 = self.down2(x2)
        # 输入的特征图尺寸为[4, 256, 120,120];  输出为[4, 512, 60,60]
        x4 = self.down3(x3)
        # 输入的特征图尺寸为[4, 512, 60,60];  输出为[4, 1024, 30,30]
        x5 = self.down4(x4)

        # 输入的特征图尺寸为[4, 1024, 30,30];  输出为[4, 512, 60, 60]
        x = self.up1(x5, x4)
        # 输入的特征图尺寸为[4, 512, 60,60];  输出为[4, 256, 120, 120]
        x = self.up2(x, x3)
        # 输入的特征图尺寸为[4, 256, 120,120];  输出为[4, 128, 240, 240]
        x = self.up3(x, x2)
        # 输入的特征图尺寸为[4, 128, 240,240];  输出为[4, 64, 480, 480]
        x = self.up4(x, x1)
        # 输入的特征图尺寸为[4, 64, 480,480];  输出为[4, 2, 480, 480]
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

        self.head1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1, bias=False), torch.nn.ReLU(),
        )

        self.mlp_mask = torch.nn.Conv2d(64, 1, 1, bias=False)
        # self.tail = conv(64, n_colors, kernel_size)

    def get_2d_emb(self, batch_size, x, y, out_ch, device):  # 生成二维位置嵌入张量，用于将位置信息编码到模型中
        out_ch = int(np.ceil(out_ch / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch)).cuda()  # 计算频率因子，用于计算正弦输入的位置编码
        pos_x = torch.arange(x, device=device).type(inv_freq.type()) * 2 * np.pi / x  # 计算x和y方向上的位置编码
        pos_y = torch.arange(y, device=device).type(inv_freq.type()) * 2 * np.pi / y
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq).cuda()  # 计算正弦输入
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq).cuda()
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).cuda()  # 获取正弦输入的嵌入
        emb_y = self.get_emb(sin_inp_y).cuda()
        emb = torch.zeros((x, y, out_ch * 2), device=device)  # 创建全零张量，并填充位置编码
        emb[:, :, : out_ch] = emb_x
        emb[:, :, out_ch: 2 * out_ch] = emb_y
        return emb  # 将结果扩展为指定的批次大小

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, args, image, viewpoint_cam):

        shuffle_rgb = image.unsqueeze(0)
        pos_enc = self.get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, device="cuda")
        # 从原噪声图中提取噪声
        noise1 = self.noise_extract(viewpoint_cam.original_image.unsqueeze(0))   # (3,H,W)

        # 从清晰图像中提取mask

        x = self.head1(image)
        x = torch.cat([x, pos_enc.permute(2, 0, 1)], 0)  # 48+16=64
        x = self.head1_1(x)     # SB模块  (64,H,W)
        x = self.head1_2(x)     # mlp(64,64)
        mask = self.mlp_mask(x)
        mask = torch.sigmoid(mask)
        # print("mask",mask)
        noise = mask * (viewpoint_cam.pre_noise) + (1 - mask) * noise1.squeeze()
        noise_img = image + noise
        return noise_img, noise, mask
######################################################################
        # x = self.head1(viewpoint_cam.original_image)
        # x = torch.cat([x, pos_enc.permute(2,0,1)],0) # 48+16=64
        # x = self.head1_1(x)     # SB模块  (64,H,W)
        # x = self.head1_2(x)
        # # x = self.head1_3(x)     # TM 多头transformer
        # add_noise = self.tail(x)
        # noise = add_noise + viewpoint_cam.pre_noise
        # noise_img = image + add_noise + viewpoint_cam.pre_noise
        # return noise_img, noise
######################################################################

        # y = x
        #
        # x = self.head1(x)
        # x = self.head1_1(x)  # SB模块
        # x = self.head1_3(x)
        # group1 = self.body1_1(x)
        # group2 = self.body2_1(x)
        # group3 = self.body3_1(x)
        #
        # group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
        # group3 = self.body3_2(self.fusion3_1(torch.cat((group2, group3), 1)))
        # group1 = self.body1_2(group1)
        #
        # group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
        # group3 = self.body3_3(self.fusion3_2(torch.cat((group2, group3), 1)))
        #
        # group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))
        #
        # x = group3

        # out = self.tail(x)
        # return y - out





class VisionEncoder(nn.Module):
    def __init__(
            self,
            img_H,
            img_W,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False,
            no_residual=False
    ):
        super(VisionEncoder, self).__init__()

        assert embedding_dim % num_heads == 0
        # assert img_H % patch_dim == 0
        # assert img_W % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_W = img_W
        self.img_H = img_H
        self.pos_every = pos_every
        self.num_patches = int((img_H // patch_dim) * (img_W // patch_dim))
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels  

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos
        self.no_residual = no_residual

        if self.mlp == False: 
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )       # 两个线性层和两个 Dropout 层，用于构建一个简单的 MLP 结构

        encoder_layer = TransformerEncoderLayer(patch_dim, embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)

        self.encoder = TransformerEncoder(encoder_layer, num_layers, self.no_residual)  # 编码层


        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, con=False, mask=None):
        x = x.unsqueeze(0)  # (1,48,H,W)                    #
        # shape == (time, B, d_model)  将图像分割成不重叠的小块 (小块的数量，批次大小，C * patch_dim * patch_dim)
        # Transformer 通常接受的输入形状是 (sequence_length, batch_size, feature_dimension)
        #  (小块的数量，批次大小，C * patch_dim * patch_dim)
        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()  # shape == (time, B, d_model)

        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x    #将输入张量 x 经过一个线性变换和 dropout 处理后，再加上原始输入
            # query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            pass
            # query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)     # 位置编码

        if self.pos_every:
            x = self.encoder(x, pos=pos, mask=mask)     # 多头注意力编码，加位置编码
            # x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x, mask)        # 多头注意力编码，无位置编码
            # x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos, mask)     # 在编码器层中应用位置编码
            # x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x        # 线性层 + 残差

        # 将输出转换为图像的形状
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:     # 如果需要返回额外的内容，则返回内容
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_H, self.img_W), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        # 返回编码后的图像表示
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), [int(self.img_H), int(self.img_W)], self.patch_dim,
                                     stride=self.patch_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        # self.register_buffer(
        #     "position_ids", torch.arange(self.seq_length).expand((1, -1))
        # )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(x.size(0)).expand((1, -1)).cuda()
            # position_ids = self.position_ids[:, : self.seq_length]  # self.position_ids???????
######################################################################################
        position_embeddings = self.pe(position_ids).clone().detach()
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, no_residual=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)    # 克隆transformer编码层，num_layers次
        self.num_layers = num_layers
        self.no_residual = no_residual

    def forward(self, src, pos=None, mask=None):
        output = src
        # 如果禁用残差连接或者层数少于4
        if self.no_residual or len(self.layers) < 4:
            for layer in self.layers:
                output = layer(output, pos=pos, mask=mask)  # 逐层应用编码器层，将结果存储在 output 中。
        else:  # encoder use residual struct 使用残差结构
            layers = iter(self.layers)  # 逐层遍历存储在 self.layers 中的多个编码器层，可以使用 next() 函数来逐个获取 self.layers 中的元素

            output1 = next(layers)(output, pos=pos, mask=mask)
            output2 = next(layers)(output1, pos=pos, mask=mask)
            output3 = next(layers)(output2, pos=pos, mask=mask)
            output4 = next(layers)(output3, pos=pos, mask=mask)
            output = output + output1 + output2 + output3 + output4

            for layer in layers:        # 继续遍历剩余的编码器层，并将它们应用于累加后的输出 output。
                output = layer(output, pos=pos, mask=mask)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, patch_dim, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.patch_dim = patch_dim

        # multihead attention 多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)

        # Implementation of Feedforward model 前馈模型
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None, mask=None):
        src2 = self.norm1(src)                  # 归一化
        q = k = self.with_pos_embed(src2, pos)  # 将图像和位置编码相加

        src2 = self.self_attn(q, k, src2)       # QKV注意力机制

        src = src + self.dropout1(src2[0])      # 对输出进行残差连接和 Dropout 处理。
        src2 = self.norm2(src)                  # 归一化
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))  # 将归一化后的张量通过前馈神经网络进行处理。
        src = src + self.dropout2(src2)         # 残差
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
