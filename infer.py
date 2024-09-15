import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from models import RTN
from baseline.dualprnet import DualPRNet
from baseline.voxelmorph import VoxelMorph
from baseline.TransMorph import CONFIGS as CONFIGS_TM
import baseline.TransMorph as TransMorph
from baseline.lapirn import LapIRN
from baseline.XMorpher import Head
import random
from medpy.metric.binary import hd95, dc

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

same_seeds(24)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def save_img_to_nii(img, data_path, data_type, data_name):
    img_np = img.detach().cpu().numpy().squeeze()
    save_path = os.path.join(data_path, data_type)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_data = nib.Nifti1Image(img_np, np.eye(4))
    
    save = os.path.join(save_path, data_name + '.nii')
    
    nib.save(image_data, save)


def main():

    data_path = '/data/XiaolongWu/miccai24/data/LPBA40_delineation/LPBA40_delineation'
    save_path = '/data/XiaolongWu/miccai24/result/IXI/RTF'
    data_name = 'Mindboggle'
    model_name = 'Dualprnet'
    iters = 5
    kernel_size = 7
    disp = False
    # img_size = (128,128,128)
    img_size = (160,192,160)
    if data_name == 'Brain':
        img_size = (160,192,160)
    elif data_name == 'Liver':
        img_size = (128,128,128)
    elif data_name == 'Feta':
        img_size = (256,256,256)
    elif data_name == 'IXI':
        img_size = (160,192,224)
    elif data_name == 'Mindboggle':
        img_size = (192,192,192)
    
    if 'RTF' in model_name:
        model = RTN(img_size, iters= iters, kernel_size=kernel_size, diffeomorphic=disp)
    elif 'VM' in model_name:
        model = VoxelMorph()
    elif 'Dualprnet' in model_name:
        model = DualPRNet(img_size)
    elif 'TransMorph' in model_name:
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
    elif 'Lapirn' in model_name:
        model = LapIRN()
    elif 'XMorpher' in model_name:
        model = Head(inshape=img_size)

    best_model = torch.load('/data/XiaolongWu/miccai24/baseline/experiments/Dualprnet_data_Mindboggle(3)(5)_ncc_1_diffusion_1_lr_0.0001_15/dsc0.372.pth.tar')['state_dict']
    model = nn.DataParallel(model)
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    if data_name == 'Brain':

        val_composed = transforms.Compose([
                                       trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.LPBABrainInferDatasetS2S(os.path.join(data_path, 'test'), transforms=val_composed)
    elif data_name == 'Liver':

        val_composed = transforms.Compose([
                                       trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.SliverInferDatasetS2S('/data/XiaolongWu/miccai24/data/sliver_val', transforms=val_composed)
        
    elif data_name == 'Feta':
        val_composed = transforms.Compose([
                                       trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.FetaInferDataset('/data/XiaolongWu/miccai24/data/feta_2.2', transforms=val_composed)
    elif data_name == 'IXI':
        val_composed = transforms.Compose([
                                       trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16))])
        val_dir = '/data/XiaolongWu/miccai24/data/IXI_data/Val/'
        atlas_dir = '/data/XiaolongWu/miccai24/data/IXI_data/atlas.pkl'
        test_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    elif data_name == 'Mindboggle':
        val_composed = transforms.Compose([
                                        trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16))])
        test_set = datasets.InferMindbooggle(data_path='/data/XiaolongWu/miccai24/data/Mindboggle/Mindboggle_3_1_1/Mindboggle_val', transforms=val_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_hd95 = AverageMeter()
    eval_hd95_raw = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_def, flow = model(x,y)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_det(flow.detach().cpu().numpy())
            eval_det.update(jac_det / np.prod(tar.shape), x.size(0))
            dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            hd95_raw = hd95(x_seg.detach().cpu().numpy(), y_seg.detach().cpu().numpy())
            hd95_ = hd95(def_out.detach().cpu().numpy(), y_seg.detach().cpu().numpy())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            eval_hd95.update(hd95_.item(), x.size(0))
            eval_hd95_raw.update(hd95_raw.item(), x.size(0))
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('Deformed HD95: {:.3f} +- {:.3f}, Affine HD95: {:.3f} +- {:.3f}'.format(eval_hd95.avg,
                                                                                    eval_hd95.std,
                                                                                    eval_hd95_raw.avg,
                                                                                    eval_hd95_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

