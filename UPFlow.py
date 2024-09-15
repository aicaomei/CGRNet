import torch.nn as nn
import torch
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class FlowEstimatorDense_temp(nn.Module):

    def __init__(self, ch_in, f_channels=(32, 32, 16, 8), ch_out=2):
        super(FlowEstimatorDense_temp, self).__init__()
        N = 0
        ind = 0
        N = ch_in
        self.conv1 = nn.Conv3d(N, f_channels[ind], kernel_size=3,  padding='same')
        # N += f_channels[ind]

        ind += 1
        self.conv2 = nn.Conv3d(f_channels[ind-1], f_channels[ind], kernel_size=3,  padding='same')
        # N += f_channels[ind]

        ind += 1
        self.conv3 = nn.Conv3d(f_channels[ind-1], f_channels[ind], kernel_size=3,  padding='same')
        # N += f_channels[ind]

        ind += 1
        self.conv4 = nn.Conv3d(f_channels[ind-1], f_channels[ind], kernel_size=3,  padding='same')
        # N += f_channels[ind]

        # ind += 1
        # self.conv5 = nn.Conv3d(N, f_channels[ind], kernel_size=3, padding='same')
        # N += f_channels[ind]
        self.num_feature_channel = N
        self.conv_last = nn.Conv3d(f_channels[ind], ch_out, kernel_size=3, padding='same')

    def forward(self, x):
        # x1 = torch.cat([self.conv1(x), x], dim=1)
        # x2 = torch.cat([self.conv2(x1), x1], dim=1)
        # x3 = torch.cat([self.conv3(x2), x2], dim=1)
        # x4 = torch.cat([self.conv4(x3), x3], dim=1)
        # x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x_out = self.conv_last(x4)
        return x4, x_out

def upsample3d_flow_as(inputs, target_as, mode="trilinear", if_rate=False):
    _, _, d, h, w = target_as.size()

    target_size = (d, h, w) 

    res = F.interpolate(inputs, size=target_size, mode=mode, align_corners=True)

    if if_rate:
        _, _, d_, h_, w_ = inputs.size()
        u_scale = (w / w_)
        v_scale = (h / h_)
        w_scale = (d / d_)
        
        u, v, w = res.chunk(3, dim=1)
        u = u.detach()  # 或者 u = u.detach()
        v = v.detach()  # 或者 v = v.detach()
        w = w.detach()  # 或者 w = w.detach()
        
        u *= u_scale
        v *= v_scale
        w *= w_scale
        
        res = torch.cat([u, v, w], dim=1)

    return res

class sgu_model(nn.Module):
    def __init__(self, image_size = (160, 192, 160)):
        super(sgu_model, self).__init__()

        

        f_channels_es = (32, 32, 16, 8)
        in_C = 32
        self.image_size = [s for s in image_size]
        self.warping_layer = SpatialTransformer(self.image_size)
        self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=4)
        self.upsample_output_conv = nn.Sequential(nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
                                                  nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
                                                  nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                                                  nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1))

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        _, _, d, h, w = flow_init.shape
        _, _, d_f, h_f, w_f = feature_1.shape
        if h != h_f or w != w_f or d != d_f:
            flow_init = upsample3d_flow_as(flow_init, feature_1, mode="trilinear", if_rate=True)
        feature_2_warp = self.warping_layer(feature_2, flow_init)
        input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
        _, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, :3, :, :, :]
        inter_mask = x_out[:, 3, :, :, :]
        inter_mask = torch.unsqueeze(inter_mask, 1)
        inter_mask = torch.sigmoid(inter_mask)
        if output_level_flow is not None:
            inter_flow = upsample3d_flow_as(inter_flow, output_level_flow, mode="trilinear", if_rate=True)
            inter_mask = upsample3d_flow_as(inter_mask, output_level_flow, mode="trilinear")
            flow_init = output_level_flow
        flow_up = self.warping_layer(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
        return flow_init, flow_up, inter_flow, inter_mask

    def output_conv(self, x):
        return self.upsample_output_conv(x)


if __name__=='__main__':
    size_fea = (1, 16, 160, 192, 160)
    size_flow = (1, 3, 80, 96, 80)
    flow_init = torch.ones(size_flow).cuda()
    feature_1 = torch.ones(size_fea).cuda()
    feature_2 = torch.ones(size_fea).cuda()
    
    model = sgu_model(image_size=size_fea[2:]).cuda()
    
    flow_init, flow_up, inter_flow, inter_mask = model(flow_init, feature_1, feature_2)
    print(flow_up.size)
    
    