import glob
from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from baseline.dualprnet import DualPRNet
from baseline.voxelmorph import VoxelMorph
from baseline.TransMorph import CONFIGS as CONFIGS_TM
import baseline.TransMorph as TransMorph
from baseline.lapirn import LapIRN
from baseline.XMorpher import Head
from natsort import natsorted
from models import RTN
import random
import argparse

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
    torch.backends.cudnn.benchmark = True
same_seeds(24)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='RTF_our', help='RTF_ET_DC_, VM, Dualprnet')
    parser.add_argument('--data_type', type=str, default='Mindboggle', help='Brain, Liver, IXI')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--data_path', type=str, default='/data/XiaolongWu/miccai24/data/LPBA40_delineation/LPBA40_delineation')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--reg_loss', type=float, default=1)
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    data_path = args.data_path
    weights = [1, args.reg_loss]  # loss weights
    lr = args.lr
    # stages = 1
    save_dir = '{}_{}({})({})_ncc_{}_diffusion_{}_lr_{}_{}/'.format(args.model_type, args.data_type, args.iters, args.kernel_size ,weights[0], weights[1], lr, args.epoch)
    if not os.path.exists('/data/XiaolongWu/miccai24/baseline/experiments/' + save_dir):
        os.makedirs('/data/XiaolongWu/miccai24/baseline/experiments/' + save_dir)
    if not os.path.exists('/data/XiaolongWu/miccai24/baseline/logs/' + save_dir):
        os.makedirs('/data/XiaolongWu/miccai24/baseline/logs/' + save_dir)
    sys.stdout = Logger('/data/XiaolongWu/miccai24/baseline/logs/' + save_dir)
    f = open(os.path.join('/data/XiaolongWu/miccai24/baseline/logs/'+save_dir, 'losses and dice' + ".txt"), "w")

    epoch_start = 0
    max_epoch = args.epoch
    if args.data_type == 'Brain':
        img_size = (160,192,160)
    elif args.data_type == 'Liver':
        img_size = (128,128,128)
    elif args.data_type == 'Feta':
        img_size = (256,256,256)
    elif args.data_type == 'IXI':
        img_size = (160,192,224)
    elif args.data_type == 'Mindboggle':
        img_size = (192,192,192)
    cont_training = False

    '''
    Initialize model
    '''
    if 'RTF' in args.model_type:
        model = RTN(img_size, iters= args.iters, kernel_size=args.kernel_size, diffeomorphic=args.disp)
    elif 'VM' in args.model_type:
        model = VoxelMorph()
    elif 'Dualprnet' in args.model_type:
        model = DualPRNet(img_size)
    elif 'TransMorph' in args.model_type:
        config = CONFIGS_TM['TransMorph']
        model = TransMorph.TransMorph(config)
    elif 'Lapirn' in args.model_type:
        model = LapIRN()
    elif 'XMorpher' in args.model_type:
        model = Head(inshape=img_size)

    if torch.cuda.device_count() > 1:
        print("使用多个 GPU。")
        model = nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.cuda(device)

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = '/data/XiaolongWu/miccai24/experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print(natsorted(os.listdir(model_dir))[-1])
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    
    if args.data_type == 'Brain':
        train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

        val_composed = transforms.Compose([
                                       trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.LPBABrainDatasetS2S(os.path.join(data_path, 'train'), transforms=train_composed)
        val_set = datasets.LPBABrainInferDatasetS2S(os.path.join(data_path, 'test'), transforms=val_composed)
    elif args.data_type == 'Liver':
        train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

        val_composed = transforms.Compose([
                                       trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.SliverDatasetS2S('/data/XiaolongWu/miccai24/data/sliver_val', transforms=train_composed)
        val_set = datasets.SliverInferDatasetS2S('/data/XiaolongWu/miccai24/data/sliver_val', transforms=val_composed)
        
    elif args.data_type == 'Feta':
        train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

        val_composed = transforms.Compose([
                                       trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.FetaDatasetS2S('/data/XiaolongWu/miccai24/data/feta_2.2', transforms=train_composed)
        val_set = datasets.FetaInferDataset('/data/XiaolongWu/miccai24/data/feta_2.2', transforms=val_composed)
    elif args.data_type == 'IXI':
        train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

        val_composed = transforms.Compose([
                                       trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16))])
        train_dir = '/data/XiaolongWu/miccai24/data/IXI_data/Train/'
        val_dir = '/data/XiaolongWu/miccai24/data/IXI_data/Val/'
        atlas_dir = '/data/XiaolongWu/miccai24/data/IXI_data/atlas.pkl'
        train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
        val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    elif args.data_type == 'Mindboggle':
        train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

        val_composed = transforms.Compose([
                                        trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.Mindbooggle(data_path='/data/XiaolongWu/miccai24/data/Mindboggle/Mindboggle_3_1_1/Mindboggle_train', transforms=train_composed)
        val_set = datasets.InferMindbooggle(data_path='/data/XiaolongWu/miccai24/data/Mindboggle/Mindboggle_3_1_1/Mindboggle_val', transforms=val_composed)
        
    
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='/data/XiaolongWu/miccai24/baseline/logs/'+save_dir)
    idx = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        for data in train_loader:
            idx += batch_size
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            output = model(x,y)
            # output = output[0:1]+output[2:]
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx/batch_size, len(train_loader)*max_epoch, loss.item(), loss_vals[0].item(), loss.item()-loss_vals[0].item()))
            
            if idx%100 == 0 or idx == len(train_loader)*max_epoch - 1:
                '''
                Validation
                '''
                eval_dsc = utils.AverageMeter()
                eval_jd = utils.AverageMeter()
                eval_dsc_raw = utils.AverageMeter()
                with torch.no_grad():
                    for data in val_loader:
                        model.eval()
                        data = [t.cuda() for t in data]
                        x = data[0]
                        y = data[1]
                        x_seg = data[2]
                        y_seg = data[3]
                        output = model(x,y)
                        def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                        dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                        dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
                        # dsc = utils.dice_val_7VOI(def_out.long(), y_seg.long())
                        # dsc_raw = utils.dice_val_7VOI(x_seg.long(), y_seg.long())
                        # dsc = utils.mask_metrics(def_out.long(), y_seg.long())
                        # dsc_raw = utils.mask_metrics(x_seg.long(), y_seg.long())
                        eval_dsc.update(dsc.item(), x.size(0))
                        eval_dsc_raw.update(dsc_raw.item(), x.size(0))
                        flow = output[1].detach().float().cpu()
                        flow_numpy = flow.numpy()
                        jd = utils.jacobian_det(flow_numpy)
                        tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                        eval_jd.update(jd / np.prod(tar.shape), x.size(0))
                        # print(eval_dsc.avg)
                        print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc.item(), dsc_raw.item()))
                best_dsc = max(eval_dsc.avg, best_dsc)
                
                writer.add_scalar('Loss/train', loss_all.avg, idx/100)
                print('{} Epoch {} loss {:.4f}'.format(save_dir, idx/100, loss_all.avg))
                print('Epoch {} loss {:.4f}'.format(idx/100, loss_all.avg), file=f, end=' ')
                print('dsc {} dsc_raw{} jd {}'.format(eval_dsc.avg, eval_dsc_raw.avg, eval_jd.avg))
                print('dsc {} dsc_raw{} jd {}'.format(eval_dsc.avg, eval_dsc_raw.avg, eval_jd.avg), file=f)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir='/data/XiaolongWu/miccai24/baseline/experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    # model_lists = natsorted(glob.glob(save_dir + '*'))
    # while len(model_lists) > max_model_num:
    #     os.remove(model_lists[0])
    #     model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)

    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()