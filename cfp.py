import torch
import torch.nn as nn
from typing import Dict, Generator, List, Optional, Tuple, Union
from im2grid import CoTr

def coords_grid_3d(batch, d, ht, wd, device='cuda'):
    """
    生成3D坐标网格
    """
    grid_d, grid_ht, grid_wd = torch.meshgrid(torch.arange(d), torch.arange(ht), torch.arange(wd))
    coords = torch.stack([grid_d, grid_ht, grid_wd], dim=-1).float().to(device)
    coords = coords.view(1, -1, 3).expand(batch, -1, -1)
    return coords

class CFP(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.self_corr = nn.Linear(c_dim, c_dim)
        # self.threshold = nn.Parameter(torch.tensor(100, dtype=torch.float32), requires_grad=True)
        
    def fetch_mask(self, self_corr, corr, thres):
        thres = 0
        corr_mask = torch.max(corr, dim=-1)[0]
        confidence = torch.zeros_like(corr)  
        confidence[corr <= thres] = -1
        confidence = confidence.unsqueeze(1)  
        self_corr = self_corr + confidence  
        
        self_corr = torch.softmax(self_corr, dim=-1)
        corr_mask[corr_mask > thres] = 1.0
        return self_corr, corr_mask.unsqueeze(-1)
    
    def forward(self, inp=None, corr_sm=None, self_corr=None, thres=0.4):
        if self_corr is None:
            batch, c, d, h, w = inp.shape
            inp = inp.reshape(batch, c, d * h * w).permute(0, 2, 1).contiguous()
            inp = self.self_corr(inp)
            self_corr = (inp * (c ** -0.5)) @ inp.transpose(1, 2)
        
        flow_attn, conf = self.fetch_mask(self_corr, corr_sm, thres=thres)
        
        return flow_attn, conf, self_corr
    
def corr(fmap1, fmap2):
    batch, c, d, h, w = fmap1.shape
    fmap1 = fmap1.view(batch, c, d * h * w)
    fmap2 = fmap2.view(batch, c, d * h * w)
    
    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, d, h, w, 1, d, h, w)
    
    corr /= torch.sqrt(torch.tensor(c).float())
    
    return corr

if __name__=='__main__':
    image1 = torch.randn(1, 32, 40, 48, 40).cuda()
    b, c, d, h, w = image1.shape
    image2 = torch.randn(1, 32, 40, 48, 40).cuda()
    inp = torch.randn(1, 32, 40, 48, 40).cuda()
    cfp = CFP(32).cuda()
    cotr = CoTr().cuda()
    flow_ = nn.Conv3d(30, 3, kernel_size=3, padding=1).cuda()
    correlation_matrix , flo = cotr(image1.permute(0,2,3,4,1), image2.permute(0,2,3,4,1))
    self_corr, _=cotr(inp.permute(0,2,3,4,1), inp.permute(0,2,3,4,1))
    
    flow_attn, conf, self_corr = cfp(self_corr=self_corr.squeeze(2), corr_sm=correlation_matrix.squeeze(2))

    flow_attn = flow_attn.reshape(b, d, h, w, 27).permute(0, 4, 1, 2, 3).contiguous()
    flo_a = flo.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
    
    flo_a = flow_(torch.cat([flow_attn.cuda(), flo_a.cuda()], dim=1))
    
    flo = flo.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, 3)
    flo_a = flo_a.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, 3)
    flo = conf * flo + (1 - conf) * (flo_a)
    flo_0 = flo.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
    
    print("Correlation Matrix:")