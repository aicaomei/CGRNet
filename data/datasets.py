import os, glob
import torch, sys
from torch.utils.data import Dataset
from torchvision import transforms
# import trans
# from .data_utils import pkload
import tifffile as tif
import nibabel as nib
from natsort import natsorted
import numpy as np


def generate_pairs(files):
    pairs = []
    for i, d1 in enumerate(files):
        for j, d2 in enumerate(files):
            if i != j:
                pairs.append([d1, d2])
    return pairs

class LPBABrainDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.files = natsorted(self.files)
        self.pairs = generate_pairs(self.files)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        index = index % len(self.pairs)
        data1_path, data2_path = self.pairs[index]
        data1 = nib.load(data1_path)
        data2 = nib.load(data2_path)
        image1 = data1.get_fdata()
        image2 = data2.get_fdata()
        
        image1, image2 = image1[None, ...], image2[None, ...]
        
        image1,image2 = self.transforms([image1, image2]) 
        
        image1, image2 = torch.from_numpy(image1), torch.from_numpy(image2)

        return image1, image2

    def __len__(self):
        return len(self.pairs)

def generate_pairs_val(files):
    pairs = []
    labels = []
    label_path = '/data/XiaolongWu/miccai24/data/LPBA40_delineation/LPBA40_delineation/label'
    for i in range(len(files) - 1):
        d1 = files[i]
        d2 = files[i + 1]
        pairs.append([d1, d2])
        labels.append([os.path.join(label_path, d1.split('/')[8].split('.')[0] + '.delineation.structure.label.nii.gz'),
                        os.path.join(label_path, d2.split('/')[8].split('.')[0] + '.delineation.structure.label.nii.gz')])
    return pairs, labels

class LPBABrainInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.files = natsorted(self.files)
        self.pairs, self.labels = generate_pairs_val(self.files)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        index = index % len(self.pairs)
        data1_path, data2_path = self.pairs[index]
        seg1_path, seg2_path = self.labels[index]
        
        data1 = nib.load(data1_path)
        data2 = nib.load(data2_path)
        
        seg1 = nib.load(seg1_path)
        seg2 = nib.load(seg2_path)
        
        image1 = data1.get_fdata()
        image2 = data2.get_fdata()

        label1 = seg1.get_fdata()
        label2 = seg2.get_fdata()
        
        image1, image2 = image1[None, ...], image2[None, ...]
        label1, label2= label1[None, ...], label2[None, ...]
        
        image1, label1 = self.transforms([image1, label1])
        image2, label2 = self.transforms([image2, label2])
        
        image1, image2, label1, label2 = torch.from_numpy(image1), torch.from_numpy(image2), torch.from_numpy(label1), torch.from_numpy(label2)
        
        return image1, image2, label1, label2

    def __len__(self):
        return len(self.pairs)

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class SliverDatasetS2S(Dataset):
    def __init__(self, data_path, transforms = None):
        self.size = [128, 128, 128]
        self.paths = data_path
        self.files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.files = natsorted(self.files)
        self.files = self.files[:15]
        self.pairs = self.generate_pairs_list()
        self.transforms = transforms
        
    def generate_pairs_list(self):
        pairs = []
        for i, d1 in enumerate(self.files):
            for j, d2 in enumerate(self.files):
                if i != j:
                    pairs.append([os.path.join(d1, 'volume.tif'), os.path.join(d2, 'volume.tif')])
        return pairs
    
    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.pairs)

        
    def __getitem__(self, index):
        index = index % len(self.pairs)
        data1_path, data2_path = self.pairs[index]
        
        image1 = tif.imread(data1_path)[np.newaxis]
        image2 = tif.imread(data2_path)[np.newaxis]
        
        image1 = torch.from_numpy(image1).float() / 255.0
        image2 = torch.from_numpy(image2).float() / 255.0
        
        
        
        return image1, image2

class SliverInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms = None):
        self.size = [128, 128, 128]
        self.paths = data_path
        self.files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.files = natsorted(self.files)
        self.files = self.files[15:]
        self.pairs, self.labels = self.generate_pairs_list()
        self.transforms = transforms
    
    
    def generate_pairs_list(self):
        pairs = []
        labels = []
        for i, d1 in enumerate(self.files):
            for j, d2 in enumerate(self.files):
                if i != j:
                    pairs.append([os.path.join(d1, 'volume.tif'), os.path.join(d2, 'volume.tif')])
                    labels.append([os.path.join(d1, 'segmentation.tif'), os.path.join(d2, 'segmentation.tif')])
        return pairs, labels
    
    
    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        data1, data2 = self.pairs[index]
        seg1, seg2 = self.labels[index]
        
        image1 = tif.imread(data1)[np.newaxis]
        image2 = tif.imread(data2)[np.newaxis]
        
        label1 = tif.imread(seg1)[np.newaxis]
        label2 = tif.imread(seg2)[np.newaxis]
        
        image1 = torch.from_numpy(image1).float() / 255.0
        image2 = torch.from_numpy(image2).float() / 255.0

        label1 = torch.from_numpy(label1).float()
        label2 = torch.from_numpy(label2).float()
        
        return image1, image2, label1, label2
        

    
class Mindbooggle(Dataset):
    def __init__(self,data_path, transforms = None):
        super().__init__()
        self.size = [192, 192, 192]
        self.paths = data_path
        self.files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.files = natsorted(self.files)
        self.pairs = self.generate_pairs_list()
        self.transforms = transforms
        
    def generate_pairs_list(self):
        pairs = []
        for i, d1 in enumerate(self.files):
            for j, d2 in enumerate(self.files):
                if i != j and d1.split('/')[8].split('-')[0] == d2.split('/')[8].split('-')[0]:
                    pairs.append([os.path.join(d1, 'data.nii.gz'), os.path.join(d2, 'data.nii.gz')])
        return pairs
    
    def __len__(self):
        return len(self.pairs)

    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())
    
    def __getitem__(self, index):
        index = index % len(self.pairs)
        data1_path, data2_path = self.pairs[index]
        data1 = nib.load(data1_path)
        data2 = nib.load(data2_path)
        image1 = data1.get_fdata()
        image2 = data2.get_fdata()
        image1 = self.normalize(image1)
        image2 = self.normalize(image2)
        
        image1, image2 = image1[None, ...], image2[None, ...]
        
        image1,image2 = self.transforms([image1, image2]) 
        
        image1, image2 = torch.from_numpy(image1), torch.from_numpy(image2)

        return image1, image2

from scipy.ndimage import zoom
class InferMindbooggle(Dataset):   
    def __init__(self,data_path, transforms = None):
        super().__init__()
        self.size = [192, 192, 192]
        self.paths = data_path
        self.files = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        self.files = natsorted(self.files)
        self.pairs, self.labels = self.generate_pairs_val()
        self.transforms = transforms
    
    def generate_pairs_val(self):
        pairs = []
        labels = []
        for i in range(len(self.files) - 1):
            d1 = self.files[i]
            d2 = self.files[i + 1]
            pairs.append([os.path.join(d1, 'data.nii.gz'), os.path.join(d2, 'data.nii.gz')])
            labels.append([os.path.join(d1, 'seg.nii.gz'), os.path.join(d2, 'seg.nii.gz')])
        return pairs, labels

    def __len__(self):
        return len(self.pairs)
    
    def Cropimg(self, image):
        target_shape = [192, 192, 192]
        current_shape = image.shape

        pad_widths = []
        for i in range(len(current_shape)):
            total_pad = max(0, target_shape[i] - current_shape[i])
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_widths.append((pad_before, pad_after))
        
        padded_image = np.pad(image, pad_widths, mode='constant', constant_values=0)
        
        zoom_factors = [target_shape[i] / float(padded_image.shape[i]) for i in range(len(target_shape))]
        resized_image = zoom(padded_image, zoom_factors, order=1)
        
        return resized_image

    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())
    
    def __getitem__(self, index):
        index = index % len(self.pairs)
        data1_path, data2_path = self.pairs[index]
        seg1_path, seg2_path = self.labels[index]
        
        data1 = nib.load(data1_path)
        data2 = nib.load(data2_path)
        
        seg1 = nib.load(seg1_path)
        seg2 = nib.load(seg2_path)
        
        image1 = self.Cropimg(data1.get_fdata())
        image2 = self.Cropimg(data2.get_fdata())
        image1 = self.normalize(image1)
        image2 = self.normalize(image2)

        label1 = self.Cropimg(seg1.get_fdata())
        label2 = self.Cropimg(seg2.get_fdata())   
        
        image1, image2 = image1[None, ...], image2[None, ...]
        label1, label2= label1[None, ...], label2[None, ...]
        
        image1, label1 = self.transforms([image1, label1])
        image2, label2 = self.transforms([image2, label2])
        
        image1, image2, label1, label2 = torch.from_numpy(image1), torch.from_numpy(image2), torch.from_numpy(label1), torch.from_numpy(label2)
        
        return image1, image2, label1, label2

    