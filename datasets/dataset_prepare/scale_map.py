import json
from matplotlib import pyplot as plt
import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import numpy as np
import pdb
from  tqdm import  tqdm
import cv2 as cv
from datasets.dataset_prepare.models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

# for VD
mean_std = ([0.42968395352363586, 0.4371049106121063, 0.4219788610935211], [
            0.23554939031600952, 0.2325684279203415, 0.23559504747390747])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])

pil_to_tensor = standard_transforms.ToTensor()
model_path = './pretrained_scale_prediction_model.pth'


def main(dataset = None):
    if dataset == 'QNRF':
        dataRoot = '/media/D/GJY/ht/ProcessedData/QNRF/'
    if dataset == 'SHHB':
        dataRoot = '/media/D/GJY/ht/ProcessedData/SHHB/'
    if dataset == 'SHHA':
        dataRoot = '/media/D/GJY/ht/ProcessedData/SHHA/'
    img_path = os.path.join(dataRoot, 'images')
    dst_size_map_path = os.path.join(dataRoot, 'size_map')
    if not os.path.exists(dst_size_map_path):
        os.makedirs(dst_size_map_path)

    file_list = os.listdir(img_path)
    print(file_list)

    GPU_ID = '0,1'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    torch.backends.cudnn.benchmark = True

    net = CrowdCounter(GPU_ID, 'Res50_SCAR')
    net.cuda()
    net.load_state_dict(torch.load(model_path), strict=False)
    net.eval()

    gen_list = tqdm(file_list)
    for fname in gen_list:

        imgname = os.path.join(img_path, fname)
        size_map_path = os.path.join(dst_size_map_path, fname.split('.')[0] + '.jpg')
        if os.path.exists(size_map_path):
            continue
        else:
            img = Image.open(imgname)

            if img.mode == 'L':
                img = img.convert('RGB')
            img = img_transform(img)[None, :, :, :]

            with torch.no_grad():
                img = Variable(img).cuda()

                crop_imgs, crop_gt, crop_masks = [], [], []
                b, c, h, w = img.shape
                slice_h,slice_w = 768, 1024
                if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                    pred_map = net.test_forward(img).cpu()
                else:
                    assert  h % 16 == 0 and w % 16 == 0

                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])

                            mask = torch.zeros(1,1,img.size(2), img.size(3)).cpu()
                            mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                            crop_masks.append(mask)
                    crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

                    # forward may need repeatng
                    crop_preds = []
                    nz, period = crop_imgs.size(0), 8
                    for i in range(0, nz, period):
                        crop_pred = net.test_forward(crop_imgs[i:min(nz, i + period)]).cpu()
                        crop_preds.append(crop_pred)

                    crop_preds = torch.cat(crop_preds, dim=0)

                    # splice them to the original size
                    idx = 0
                    pred_map = torch.zeros(1,1,img.size(2), img.size(3)).cpu().float()

                    for i in range(0, h, slice_h):
                        h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                        for j in range(0, w, slice_w):
                            w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                            pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                            idx += 1

                    # for the overlapping area, compute average value
                    mask = crop_masks.sum(dim=0)
                    pred_map = (pred_map / mask)
                assert  pred_map[0,0].size() == img[0,0].size()
                pred_map = pred_map.data.numpy()[0, 0, :, :]

            # sio.savemat(dataRoot + '/size_mat/' +fname.split('.')[0]+'.mat',{'matrix':pred_map})
            # pred_frame = plt.gca()
            # plt.imshow(pred_map, cmap='jet')
            # pred_frame.axes.get_yaxis().set_visible(False)
            # pred_frame.axes.get_xaxis().set_visible(False)
            # pred_frame.spines['top'].set_visible(False)
            # pred_frame.spines['bottom'].set_visible(False)
            # pred_frame.spines['left'].set_visible(False)
            # pred_frame.spines['right'].set_visible(False)
            # plt.savefig(dataRoot + '/size_map/' + fname.split('.')[0]  + '.png', \
            #             bbox_inches='tight', pad_inches=0, dpi=150)
            # plt.close()

            # pdb.set_trace()
            # print(f'{filename} {pred:.4f}', file=record)
            cv.imwrite(size_map_path, pred_map, [cv.IMWRITE_JPEG_QUALITY, 100])
            # #
    # record.close()


if __name__ == '__main__':
    # main('QNRF')
    main('SHHA')
    # main('SHHB')