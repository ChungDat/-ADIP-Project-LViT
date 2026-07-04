# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Vit import VisionTransformer, Reconstruct
from nets.UNet import ConvBatchNorm, _make_nConv, get_activation, DownBlock
from nets.pixlevel import PixLevelModule

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        # self.pix_module1 = PixLevelModule(64)
        # self.pix_module2 = PixLevelModule(128)
        # self.pix_module3 = PixLevelModule(256)
        # self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x, text):
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        y1 = self.downVit(x1, x1, text1) # [4, 196, 64]
        x2 = self.down1(x1) # [4, 128, 112, 112]
        y2 = self.downVit1(x2, y1, text2) # [4, 196, 128]
        x3 = self.down2(x2) # [4, 256, 56, 56]
        y3 = self.downVit2(x3, y2, text3) # [4, 196, 256]
        x4 = self.down3(x3) # [4, 512, 28, 28]
        y4 = self.downVit3(x4, y3, text4) # [4, 196, 512]
        x5 = self.down4(x4) # [4, 512, 14, 14]
        y4 = self.upVit3(y4, y4, text4, True) # [4, 196, 512]
        y3 = self.upVit2(y3, y4, text3, True) # [4, 196, 256]
        y2 = self.upVit1(y2, y3, text2, True) # [4, 196, 128]
        y1 = self.upVit(y1, y2, text1, True) # [4, 196, 64]
        x1 = self.reconstruct1(y1) + x1 # [4, 64, 224, 224]
        x2 = self.reconstruct2(y2) + x2 # [4, 128, 112, 112]
        x3 = self.reconstruct3(y3) + x3 # [4, 256, 56, 56]
        x4 = self.reconstruct4(y4) + x4 # [4, 512, 28, 28]
        x = self.up4(x5, x4) # 
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits
