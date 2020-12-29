# -*- coding: utf-8 -*-

import os
from importlib import import_module
import misc.transforms as own_transforms
import torchvision.transforms as standard_transforms
from . import basedataset
from . import setting
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import pdb
from config import  cfg
def createTrainData(datasetname, Dataset, cfg_data):

    folder, list_file = None, None

    if datasetname in ['SHHA', 'SHHB' , 'QNRF', 'JHU', 'NWPU', 'FDST']:
        list_file=[]
        list_file.append({'data_path':cfg_data.DATA_PATH,
                          'imgId_txt': cfg_data.TRAIN_LST,
                          'box_gt_txt': []})
    else:
        print('dataset is not exist')

    main_transform = own_transforms.Compose([
        own_transforms.ScaleByRateWithMin([0.8, 1.2], cfg_data.TRAIN_SIZE[1], cfg_data.TRAIN_SIZE[0]),
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
        own_transforms.RandomHorizontallyFlip(),
    ])

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    mask_transform = standard_transforms.Compose([
        standard_transforms.ToTensor()
    ])

    train_set = Dataset(datasetname, 'train',
        main_transform = main_transform,
        img_transform = img_transform,
        mask_transform = mask_transform,
        list_file = list_file
    )
    if datasetname in ['SHHA', 'SHHB' , 'QNRF', 'JHU', 'NWPU']:
        return DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=6, shuffle=True, drop_last=True)
    elif datasetname in ['FDST']:
        train_sampler = RandomSampler(data_source=train_set, replacement=True,  num_samples=1000)
        return DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, sampler=train_sampler, num_workers=6, drop_last=True)
    else:
        return 'error'

def createValData(datasetname, Dataset, cfg_data):

    if datasetname in ['SHHA', 'SHHB' , 'QNRF', 'JHU', 'NWPU', 'FDST']:
        list_file=[]
        list_file.append({'data_path':cfg_data.DATA_PATH,
                          'imgId_txt': cfg_data.VAL_LST,
                          'box_gt_txt': cfg_data.VAL4EVAL})
    else:
        print('dataset is not exist')

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    mask_transform = standard_transforms.Compose([
        standard_transforms.ToTensor()

    ])

    val_set = Dataset(datasetname, 'val',
        img_transform = img_transform,
        mask_transform = mask_transform,
        list_file = list_file
    )

    if datasetname in ['SHHA', 'SHHB' , 'QNRF', 'JHU', 'NWPU']:
        return DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=6, shuffle=True, drop_last=True)
    elif datasetname in ['FDST']:
        val_sampler = RandomSampler(data_source=val_set, replacement=True,  num_samples=200)
        return DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, sampler=val_sampler, num_workers=6, drop_last=True)
    else:
        return 'error'


def createRestore(mean_std):
    return standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def loading_data(datasetname):
    datasetname = datasetname.upper()
    cfg_data = getattr(setting, datasetname).cfg_data

    Dataset = basedataset.Dataset        
    
    train_loader = createTrainData(datasetname, Dataset, cfg_data)
    val_loader = createValData(datasetname, Dataset, cfg_data)

    restore_transform = createRestore(cfg_data.MEAN_STD)
    return train_loader, val_loader, restore_transform

