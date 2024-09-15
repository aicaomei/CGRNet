import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SpatialTransformer

class ResBlock(nn.Module):
    def __init__(self, inputc):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(inputc, inputc, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(inputc, inputc, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.conv2(x1)
        output = x + x2

        return output


class RefineBlock(nn.Module):
    def __init__(self, inputc1, inputc2):
        super(RefineBlock, self).__init__()

        self.conv1 = nn.Conv3d(inputc1, inputc2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(inputc2, inputc2, kernel_size=3, stride=2, padding=1)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, y):
        _, _, d, h, w = x.shape
        output = self.conv1(x) + F.interpolate(y, (d, h, w), mode='trilinear', align_corners=True)
        output = self.lrelu(self.conv2(output))

        return output


class Estimator(nn.Module):
    def __init__(self, inputc):
        super(Estimator, self).__init__()
        self.conv1 = nn.Conv3d(inputc, 3, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv3d(48, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)
        # self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # x = self.lrelu(self.conv1(x))
        # x = self.lrelu(self.conv2(x))
        # x = self.lrelu(self.conv3(x))
        flow = self.conv1(x)
        return flow


class DualExt(nn.Module):
    def __init__(self):
        super(DualExt, self).__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = ResBlock(inputc=16)
        self.conv2_3 = ResBlock(inputc=16)
        self.refine2 = RefineBlock(inputc1=8, inputc2=16)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = ResBlock(inputc=32)
        self.conv3_3 = ResBlock(inputc=32)
        self.refine3 = RefineBlock(inputc1=16, inputc2=32)

        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = ResBlock(inputc=32)
        self.conv4_3 = ResBlock(inputc=32)
        self.refine4 = RefineBlock(inputc1=32, inputc2=32)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_2x = self.lrelu(self.conv1(x))
        x_2x = self.lrelu(self.conv1_2(x_2x))  # 8

        x_4x = self.lrelu(self.conv2(x_2x))
        x_4x = self.lrelu(self.conv2_2(x_4x))
        x_4x = self.lrelu(self.conv2_3(x_4x))
        x_4x = self.refine2(x_2x, x_4x)  # 16

        x_8x = self.lrelu(self.conv3(x_4x))
        x_8x = self.lrelu(self.conv3_2(x_8x))
        x_8x = self.lrelu(self.conv3_3(x_8x))
        x_8x = self.refine3(x_4x, x_8x)  # 32

        x_16x = self.lrelu(self.conv4(x_8x))
        x_16x = self.lrelu(self.conv4_2(x_16x))
        x_16x = self.lrelu(self.conv4_3(x_16x))
        x_16x = self.refine4(x_8x, x_16x)  # 32

        return {'1/2': x_2x, '1/4': x_4x, '1/8': x_8x, '1/16': x_16x}


class DualPRNet(nn.Module):
    def __init__(self, shape):
        super(DualPRNet, self).__init__()
        self.extraction = DualExt()

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.estimator_16x = Estimator(inputc=64)
        self.estimator_8x = Estimator(inputc=64)
        self.estimator_4x = Estimator(inputc=32)
        self.estimator_2x = Estimator(inputc=16)

        shape1x = [s for s in shape]
        shape2x = [s/2 for s in shape]
        shape4x = [s/4 for s in shape]
        shape8x = [s/8 for s in shape]
        self.reconstruction1x = SpatialTransformer(shape1x)
        self.reconstruction2x = SpatialTransformer(shape2x)
        self.reconstruction4x = SpatialTransformer(shape4x)
        self.reconstruction8x = SpatialTransformer(shape8x)

    def forward(self, moving, fixed):
        f_fea = self.extraction(fixed)
        m_fea = self.extraction(moving)

        # 1/16
        fm_16x = (torch.cat([f_fea['1/16'], m_fea['1/16']], 1))
        flow_16x = self.estimator_16x(fm_16x)

        # 1/8
        b, c, d, h, w = f_fea['1/8'].shape
        flow_16x_up = F.interpolate(flow_16x, size=(d, h, w), mode='trilinear', align_corners=True) * 2.0
        m_fea_8x_warped = self.reconstruction8x(m_fea['1/8'], flow_16x_up)
        fm_8x = (torch.cat([f_fea['1/8'], m_fea_8x_warped], 1))
        flow_8x = self.estimator_8x(fm_8x) + flow_16x_up

        # 1/4
        b, c, d, h, w = f_fea['1/4'].shape
        flow_8x_up = F.interpolate(flow_8x, size=(d, h, w), mode='trilinear', align_corners=True) * 2.0
        m_fea_4x_warped = self.reconstruction4x(m_fea['1/4'], flow_8x_up)
        fm_4x = (torch.cat([f_fea['1/4'], m_fea_4x_warped], 1))
        flow_4x = self.estimator_4x(fm_4x) + flow_8x_up

        # 1/2
        b, c, d, h, w = f_fea['1/2'].shape
        flow_4x_up = F.interpolate(flow_4x, size=(d, h, w), mode='trilinear', align_corners=True) * 2.0
        m_fea_2x_warped = self.reconstruction2x(m_fea['1/2'], flow_4x_up)
        fm_2x = (torch.cat([f_fea['1/2'], m_fea_2x_warped], 1))
        flow_2x = self.estimator_2x(fm_2x) + flow_4x_up

        # 1
        b, c, d, h, w = moving.shape
        flow_2x_up = F.interpolate(flow_2x, size=(d, h, w), mode='trilinear', align_corners=True) * 2.0
        moved_img = self.reconstruction1x(moving, flow_2x_up)

        # return {'flow': flow_2x_up, 'flow_2x': flow_2x, 'flow_4x': flow_4x, 'flow_8x': flow_8x, 'flow_16x': flow_16x}
        return moved_img, flow_2x_up


if __name__ == '__main__':
    size = (1, 1, 160, 192, 160)
    model = DualPRNet(size[2:]).cuda()
    # print(str(model))
    A = torch.ones(size)
    B = torch.ones(size)
    out = model(A.cuda(), B.cuda())
    print(out[0].shape, out[1].shape)
    
    
