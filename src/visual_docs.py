
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import mydataset
from LMs.HFTrainer import MyTrainer
import LMs
from OCRs import img_util

def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/train.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # params.device = torch.device('cpu')
    print('Using device:', params.device)

    # section 2, objective function and output dim/ move to trainer
    # this is usually covered by huggingface models
    mydata = mydataset.setup(params)
    print(mydata.raw_test)

    # sectioin 3: downstread use
    group1 = [101,11,36, 77, 104]
    group2 = [15, 142, 106, 46, 76]
    group3 = [16, 129]
    group4 = [25, 60]
    group5 = [45, 86]

    id_set = group1+group2+group3+group4+group5
    id_set = set(['test_' + str(id) for id in id_set])

    base_folder = '/home/ubuntu/air/vrdu/datasets/findoc_v1/images/'
    dir = '/home/ubuntu/python_projects/Spatial-LM/data'
    for sample in mydata.raw_test:
        if sample['id'] in id_set:
            file_path = os.path.join(base_folder, sample['id'] + '.jpg')
            save_path = os.path.join(dir, sample['id'] + '.jpg')
            # key function you invoke:
            img_util.visual_pic(sample, save_path=save_path)


