import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import SpatialTransformer


class VoxelMorph(nn.Module):
    def __init__(self, flow_multiplier=1.):
        super(VoxelMorph, self).__init__()
        self.flow_multiplier = flow_multiplier
        #  encoder
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)

        #  decoder
        self.decode5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.decode4 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.decode3 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.decode2 = nn.Conv3d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.decode2_1 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.decode1 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)

        self.flow = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.transform = SpatialTransformer([192,192,192])
        # self.transform = SpatialTransformer([160,192,160])
        # self.transform = SpatialTransformer([128,128,128])
        # self.transform = SpatialTransformer([160,192,224])

    def forward(self, moving, fixed):
        concatImgs = torch.cat([moving, fixed], 1)
        encode_1x = self.lrelu(self.conv1(concatImgs))        # 1 -> 1
        encode_2x = self.lrelu(self.conv2(encode_1x))         # 1 -> 1/2
        encode_4x = self.lrelu(self.conv3(encode_2x))         # 1/2 -> 1/4
        encode_8x = self.lrelu(self.conv4(encode_4x))         # 1/4 -> 1/8
        encode_16x = self.lrelu(self.conv5(encode_8x))        # 1/8 -> 1/16

        _, _, d, h, w = encode_8x.shape
        decode_16x = self.lrelu(self.decode5(encode_16x))
        decode_8x_up = F.interpolate(decode_16x, size=(d, h, w))
        decode_8x_concat = torch.cat([decode_8x_up, encode_8x], 1)  # 1/16 -> 1/8

        _, _, d, h, w = encode_4x.shape
        decode_8x = self.lrelu(self.decode4(decode_8x_concat))
        decode_4x_up = F.interpolate(decode_8x, size=(d, h, w))
        decode_4x_concat = torch.cat([decode_4x_up, encode_4x], 1)  # 1/8 -> 1/4

        _, _, d, h, w = encode_2x.shape
        decode_4x = self.lrelu(self.decode3(decode_4x_concat))
        decode_2x_up = F.interpolate(decode_4x, size=(d, h, w))
        decode_2x_concat = torch.cat([decode_2x_up, encode_2x], 1)  # 1/4 -> 1/2

        _, _, d, h, w = concatImgs.shape
        decode_2x = self.lrelu(self.decode2(decode_2x_concat))  # 1/2 -> 1/2
        decode_2x = self.lrelu(self.decode2_1(decode_2x))

        decode_1x_up = F.interpolate(decode_2x, size=(d, h, w))
        decode_1x_concat = torch.cat([decode_1x_up, encode_1x], 1)  # 1/2 -> 1
        decode_1x = self.lrelu(self.decode1(decode_1x_concat))

        net = self.flow(decode_1x)
        
        moved = self.transform(moving, net* self.flow_multiplier)

        return moved, net * self.flow_multiplier












