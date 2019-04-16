import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data

import pdb

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(data.Dataset):
    def __init__(self, datadir, tear_type, use_gpu):
        super().__init__()
        self.use_gpu = use_gpu

        label_dict = {}
        self.paths = []
        abonormal_label_dict = {}
        
        if datadir[-1]=="/":
            datadir = datadir[:-1]
        self.datadir = datadir

        for i, line in enumerate(open(datadir+'-'+tear_type+'.csv').readlines()):
            line = line.strip().split(',')
            filename = line[0]
            label = line[1]
            label_dict[filename] = int(label)

        for i, line in enumerate(open(datadir+'-'+"abnormal"+'.csv').readlines()):
            line = line.strip().split(',')
            filename = line[0]
            label = line[1]
            abnormal_label_dict[filename] = int(label)

        for filename in os.listdir(os.path.join(datadir, "axial")):
            if filename.endswith(".npy"):
                self.paths.append(filename)
        
        self.labels = [label_dict[path.split(".")[0]] for path in self.paths]
        self.abnormal_labels = [abnormal_label_dict[path.split(".")[0]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        filename = self.paths[index]
        vol_axial = np.load(os.path.join(self.datadir, "axial", filename))
        vol_sagit = np.load(os.path.join(self.datadir, "sagittal", filename))
        vol_coron = np.load(os.path.join(self.datadir, "coronal", filename))

        # axial
        pad = int((vol_axial.shape[2] - INPUT_DIM)/2)
        vol_axial = vol_axial[:,pad:-pad,pad:-pad]
        vol_axial = (vol_axial-np.min(vol_axial))/(np.max(vol_axial)-np.min(vol_axial))*MAX_PIXEL_VAL
        vol_axial = (vol_axial - MEAN) / STDDEV
        vol_axial = np.stack((vol_axial,)*3, axis=1)
        vol_axial_tensor = torch.FloatTensor(vol_axial)
        
        # sagittal
        pad = int((vol_sagit.shape[2] - INPUT_DIM)/2)
        vol_sagit = vol_sagit[:,pad:-pad,pad:-pad]
        vol_sagit = (vol_sagit-np.min(vol_sagit))/(np.max(vol_sagit)-np.min(vol_sagit))*MAX_PIXEL_VAL
        vol_sagit = (vol_sagit - MEAN) / STDDEV
        vol_sagit = np.stack((vol_sagit,)*3, axis=1)
        vol_sagit_tensor = torch.FloatTensor(vol_sagit)

        # coronal
        pad = int((vol_coron.shape[2] - INPUT_DIM)/2)
        vol_coron = vol_coron[:,pad:-pad,pad:-pad]
        vol_coron = (vol_coron-np.min(vol_coron))/(np.max(vol_coron)-np.min(vol_coron))*MAX_PIXEL_VAL
        vol_coron = (vol_coron - MEAN) / STDDEV
        vol_coron = np.stack((vol_coron,)*3, axis=1)
        vol_coron_tensor = torch.FloatTensor(vol_coron)

        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_axial_tensor, vol_sagit_tensor, vol_coron_tensor, label_tensor, self.abnormal_labels[index]

    def __len__(self):
        return len(self.paths)

def load_data(task="acl", use_gpu=False):
    train_dir = "data/train"
    valid_dir = "data/valid"
    
    train_dataset = Dataset(train_dir, task, use_gpu)
    valid_dataset = Dataset(valid_dir, task, use_gpu)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False)

    return train_loader, valid_loader
