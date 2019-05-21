'''
MRNet Challenge Submission script

Output predictions file should be a 3-column CSV file (with no header),
where each column contains a prediction for abnormality, ACL tear, and
meniscal tear, in that order

Yash Bhalgat, yashbhalgat95@gmail.com
'''
import numpy as np
import os
import pickle
import sys

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

from model import TripleMRNet

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


def get_study(axial_path, sagit_path, coron_path):
    vol_axial = np.load(axial_path)
    vol_sagit = np.load(sagit_path)
    vol_coron = np.load(coron_path)

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
    
    return {"axial": vol_axial_tensor,
            "sagit": vol_sagit_tensor,
            "coron": vol_coron_tensor}


def get_prediction(model, tensors, abnormality_prior=None):
    vol_axial = tensors["axial"].cuda()
    vol_sagit = tensors["sagit"].cuda()
    vol_coron = tensors["coron"].cuda()

    vol_axial = Variable(vol_axial)
    vol_sagit = Variable(vol_sagit)
    vol_coron = Variable(vol_coron)

    logit = model.forward(vol_axial, vol_sagit, vol_coron)
    pred = torch.sigmoid(logit)
    pred_npy = pred.data.cpu().numpy()[0][0]
    
    if abnormality_prior:
        pred_npy = pred_npy * abnormality_prior

    return pred_npy


if __name__=="__main__":
    input_csv_path = sys.argv[1]
    preds_csv_path = sys.argv[2]

    # Assuming that the input csv has all three views for each ID.
    # And that entries are sorted by ID.
    views = []
    for i, fpath in enumerate(open(input_csv_path).readlines()):
        if "axial" in fpath:
            axial_path = fpath.strip()
        elif "sagittal" in fpath:
            sagit_path = fpath.strip()
        elif "coronal" in fpath:
            coron_path = fpath.strip()
        if i%3==2:
            views.append(get_study(axial_path, sagit_path, coron_path))
    
    # Loading all models
    abnormal_model_path = "abnormal_triple_alex/val0.1071_train0.0868_epoch8"
    acl_model_path = "acl_triple_alex/val0.1310_train0.0504_epoch30"
    meniscal_model_path = "meniscal_triple_alex/val0.2645_train0.1142_epoch22"

    abnormal_model = TripleMRNet(backbone="alexnet")
    state_dict = torch.load(abnormal_model_path)
    abnormal_model.load_state_dict(state_dict)
    abnormal_model.cuda()
    abnormal_model.eval()
    
    acl_model = TripleMRNet(backbone="alexnet")
    state_dict = torch.load(acl_model_path)
    acl_model.load_state_dict(state_dict)
    acl_model.cuda()
    acl_model.eval()

    meniscal_model = TripleMRNet(backbone="alexnet")
    state_dict = torch.load(meniscal_model_path)
    meniscal_model.load_state_dict(state_dict)
    meniscal_model.cuda()
    meniscal_model.eval()

    # Getting predictions
    with open(preds_csv_path, "w") as csv_file:
        for study in views:
            abnormality = get_prediction(
                    abnormal_model,
                    study,
                    abnormality_prior=None)
            acl_tear = get_prediction(
                    acl_model,
                    study,
                    abnormality_prior=abnormality)
            meniscal_tear = get_prediction(
                    meniscal_model,
                    study,
                    abnormality_prior=abnormality)
            
            csv_file.write(",".join(
                [str(abnormality), str(acl_tear), str(meniscal_tear)]))
            csv_file.write("\n")
