import torch
import torch.nn as nn
# from mmcv.ops import deform_conv3d
import numpy as np
from torch.distributions.normal import Normal
from swinTransformer import SwinTransformer3DFPN
from UPFlow import sgu_model
# from monai.networks.nets.segresnet import SegResNet
import torch.nn.functional as F
from typing import Dict, Generator, List, Optional, Tuple, Union
from torch.nn.modules.utils import _triple

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from STencoder import SwinTransformer
import cfp
from im2grid import CoTr

def identity_grid_torch(size: Tuple[int, ...], device: Union[torch.device,str]="cuda", stack_dim: int=0) -> torch.Tensor:
    """
    Computes an identity grid for torch
    """
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids, dim=stack_dim)
    grid = grid.unsqueeze(0).float().to(device)

    return grid

def concat_flow(
    second_flow: torch.Tensor, first_flow: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    grid = identity_grid_torch(first_flow.shape[-3:], device=first_flow.device)
    new_locs = grid + second_flow

    shape = second_flow.shape[2:]

    # need to normalize grid values to [-1, 1] for resampler
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

    resampled = F.grid_sample(first_flow, new_locs, align_corners=True, mode=mode) +second_flow
    return resampled

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)



class SegResNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> torch.Tensor:
        up_x = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            up_x.append(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x, up_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x, up_x = self.decode(x, down_x)
        return  up_x[0], up_x[1], x


class Fusion(nn.Module):
    def __init__(self, inputc1):
        super(Fusion, self).__init__()

        self.conv1 = nn.Conv3d(inputc1, 48, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(48, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, inputc1, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, flows):
        x = self.lrelu(self.conv1(flows))
        x = self.lrelu(self.conv2(x))
        k = self.conv3(x)

        b, _, d, h, w = flows.shape
        flow = (flows.view(b, -1, 3, d, h, w) * F.softmax(k.view(b, -1, 3, d, h, w), 1)).sum(1)
        return flow

    

class LKAttentionModule3D(nn.Module):
    def __init__(self, dim, kernel_size_DW, padding_DW, kernel_size_DWD, padding_DWD, dilation):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_size_DW, padding=padding_DW, groups=dim, bias=True)
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=kernel_size_DWD, stride=1, padding=padding_DWD, dilation=dilation, groups=dim, bias=True)
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=1,bias=True)

    def forward(self, x):
        # u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn*x


class LKA_encoder(nn.Module):
    def __init__(self, dim, kernel_size_DW, padding_DW, kernel_size_DWD, padding_DWD, dilation, indentity):
        super(LKA_encoder, self).__init__()
        self.layer_indentity = indentity
        
        self.layer_regularKernel = nn.Conv3d(dim, dim, kernel_size = 3, stride=1, padding=1, bias=True)
        self.layer_largeKernel = LKAttentionModule3D(dim, kernel_size_DW=kernel_size_DW, padding_DW=padding_DW, kernel_size_DWD=kernel_size_DWD, padding_DWD=padding_DWD, dilation=dilation)
        self.layer_oneKernel = nn.Conv3d(dim, dim,kernel_size=1,bias=True)
        
    def forward(self, inputs):
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        if self.layer_indentity:  
            outputs = regularKernel + largeKernel + oneKernel + inputs
        else:
            outputs = regularKernel + largeKernel + oneKernel
        return outputs

class LKA_Flow(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_DW, padding_DW, kernel_size_DWD, padding_DWD, dilation):
        super(LKA_Flow, self).__init__()
        bias_opt = True
        self.norm_conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1 = LKA_encoder(64, kernel_size_DW=kernel_size_DW, padding_DW=padding_DW, kernel_size_DWD=kernel_size_DWD, padding_DWD=padding_DWD, dilation=dilation,indentity=True)
        self.norm_conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv2 = LKA_encoder(32, kernel_size_DW=kernel_size_DW, padding_DW=padding_DW, kernel_size_DWD=kernel_size_DWD, padding_DWD=padding_DWD, dilation=dilation, indentity=True)
        self.norm_conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.conv3 = LKA_encoder(16, kernel_size_DW=kernel_size_DW, padding_DW=padding_DW, kernel_size_DWD=kernel_size_DWD, padding_DWD=padding_DWD, dilation=dilation, indentity=True)
        self.norm_conv4 = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        self.conv4 = LKA_encoder(8, kernel_size_DW=kernel_size_DW, padding_DW=padding_DW, kernel_size_DWD=kernel_size_DWD, padding_DWD=padding_DWD, dilation=dilation, indentity=True)
        self.norm_conv5 = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
        self.conv5 = LKA_encoder(out_channels, kernel_size_DW=kernel_size_DW, padding_DW=padding_DW, kernel_size_DWD=kernel_size_DWD, padding_DWD=padding_DWD, dilation=dilation, indentity=True)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        x = self.lrelu(self.norm_conv1(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.norm_conv2(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.norm_conv3(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.norm_conv4(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.norm_conv5(x))
        x = self.lrelu(self.conv5(x))
        return x
    

class RTN(nn.Module):
    def __init__(self, inshape=(160,192,160), in_channel=1, channels=16, iters=12, hidden_dim=32, kernel_size=5, diffeomorphic=False):
        super(RTN, self).__init__()
        
        self.channel = channels
        self.inshape = inshape
        self.iters = iters
        self.starting = None
        self.diffeomorphic = diffeomorphic
        
        
        self.feature_extractor = SegResNet(in_channels=in_channel, out_channels= 16)
        self.context = SegResNet(in_channels=in_channel, out_channels= 16)
        
        self.cfp = cfp.CFP(16)
        self.threshold = nn.Parameter(torch.tensor(0.4, dtype=torch.float32), requires_grad=True)
        self.cotr = CoTr()
        self.atten_flow = nn.Conv3d(30, 3, kernel_size=3, padding=1)

        self.self_attention = LKA_Flow(channels*2, 16, kernel_size_DW=5, padding_DW=2, kernel_size_DWD=3, padding_DWD=3, dilation=3)
        self.cross_attention = LKA_Flow(channels*2, 16, kernel_size_DW=5, padding_DW=2, kernel_size_DWD=3, padding_DWD=3, dilation=3)
        
        self.updateLK_7 = LKA_Flow(channels*2, 3, kernel_size_DW=5, padding_DW=2, kernel_size_DWD=3, padding_DWD=3, dilation=3)

        self.upflow = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.fusion = Fusion(3*3)
        
        
        self.transformerx2 = SpatialTransformer([s/2 for s in inshape])
        self.transformer = SpatialTransformer([s for s in inshape])
        
    def apply_diffeomorphism(self, flow):
        scale = 1 / (2**7)
        flow = scale * flow
        for _ in range(7):
            flow = flow + concat_flow(flow, flow)
        return flow
    
    def forward_LK_MMA(self, moving, fixed):
        fixed_featx4 ,fixed_featx2, _ = self.feature_extractor(fixed)
        moving_featx4, moving_featx2, _ = self.feature_extractor(moving)
        context, _, _ = self.context(moving)
        
        fixed_sfx4 = self.self_attention(fixed_featx4)
        moving_sfx4 = self.self_attention(moving_featx4)
        
        fixed_cfx4 = self.cross_attention(torch.cat([fixed_sfx4, moving_sfx4],dim=1))
        moving_cfx4 = self.cross_attention(torch.cat([moving_sfx4, fixed_sfx4],dim=1))
        
        b, c, d, h, w = fixed_cfx4.shape
        correlation_matrix, flo = self.cotr(fixed_cfx4.permute(0,2,3,4,1), moving_cfx4.permute(0,2,3,4,1))
        self_corr, _=self.cotr(context.permute(0,2,3,4,1), context.permute(0,2,3,4,1))
        flow_attn, conf, _ = self.cfp(self_corr=self_corr.squeeze(2), corr_sm=correlation_matrix.squeeze(2), thres=self.threshold)
        
        flow_attn = flow_attn.reshape(b, d, h, w, 27).permute(0, 4, 1, 2, 3).contiguous()
        flo_a = flo.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
        
        flo_a = self.atten_flow(torch.cat([flow_attn, flo_a], dim=1))
        flo = flo.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, 3)
        flo_a = flo_a.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, 3)
        flo = conf * flo + (1 - conf) * (flo_a)
        flow = flo.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
        
        
        flow = self.upflow(flow) * 2.0  
        moving_featx2 = self.transformerx2(moving_featx2, flow)
        
        for _ in range(self.iters):
            delta_flow = self.updateLK_7(torch.cat([fixed_featx2, moving_featx2], dim=1)) 
            flow = self.transformerx2(flow, delta_flow) + delta_flow
            moving_featx2 = self.transformerx2(moving_featx2, delta_flow)
        
        flow = self.upflow(flow) * 2.0
        
        moved = self.transformer(moving, flow)
        
        return [moved, flow]
    

    def forward(self, moving, fixed):
        moved, flow = self.forward_LK_MMA(moving, fixed)
        
        return [moved, flow] 
        
        

if __name__ == '__main__':
    size = (1, 1, 192, 192, 192)
    model = RTN(size[2:], iters=2, kernel_size=7).cuda()
    A = torch.ones(size)
    B = torch.ones(size)
    out = model(A.cuda(), B.cuda())
    print(out[0].shape, out[1].shape)