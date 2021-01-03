# -*- coding: utf-8 -*-

import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torch
import math
import json
class Dataset(data.Dataset):
    def __init__(self, datasetname, mode, **argv):
        self.mode = mode
        self.datasetname = datasetname

        self.img_path = []
        self.json_path = []
        self.mask_path = []
        self.box_gt = []
        self.info = []

        for data_infor in argv['list_file']:
            data_path,imgId_txt,box_gt_txt = data_infor['data_path'],data_infor['imgId_txt'],data_infor['box_gt_txt']
            with open(os.path.join(data_path,imgId_txt)) as f:
                lines = f.readlines()

            box_gt_Info = []
            if self.mode == 'val':
                box_gt_Info= self.read_box_gt(os.path.join(data_path,box_gt_txt))

            if "NWPU" in data_path:
                for line in lines:
                    splited = line.strip().split()
                    self.img_path.append(os.path.join(data_path,'images',splited[0]+'.jpg'))
                    self.mask_path.append(os.path.join(data_path, 'mask_50_60', splited[0] + '.png'))
                    if self.mode == 'val':
                        self.box_gt.append(box_gt_Info[int(splited[0])])
                    self.info.append(splited[1:3]) # lum, crowd level
            elif "QNRF" in data_path or "JHU" in data_path:
                for line in lines:
                    line=line.strip()
                    self.img_path.append(os.path.join(data_path,  'images',line + '.jpg'))
                    # self.json_path.append(os.path.join(root, self.mode, 'jsons', splited[0] + '.json'))
                    self.mask_path.append(os.path.join(data_path,  'mask_30_60', line + '.png'))
                    if self.mode == 'val':
                        self.box_gt.append(box_gt_Info[int(line)])
            else:
                for line in lines:
                    line=line.strip()
                    self.img_path.append(os.path.join(data_path,  'images',line + '.jpg'))
                    # self.json_path.append(os.path.join(root, self.mode, 'jsons', splited[0] + '.json'))
                    self.mask_path.append(os.path.join(data_path,  'mask_50_60', line + '.png'))
                    if self.mode == 'val':
                        self.box_gt.append(box_gt_Info[int(line)])




        self.num_samples = len(self.img_path)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.mask_transform = None
        if 'mask_transform' in argv.keys():
            self.mask_transform = argv['mask_transform']


        if self.mode is 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} train images.')
        if self.mode is 'val':

            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images.')
        if self.mode is 'test':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} test images.')
            import json
            with open(data_path + '/res.json', 'r') as f:
                self.res = json.load(f)

    
    def __getitem__(self, index):

        img, mask_map = self.read_image_and_gt(index)
        # print(img.size, mask_map.size)
        if self.main_transform is not None:
            img, mask_map  = self.main_transform(img, mask_map)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask_map = self.mask_transform(mask_map)

        if self.mode == 'train' :

            return img, mask_map
        else:
            return img, mask_map, self.box_gt[index]
    def __len__(self):
        return self.num_samples


    def read_image_and_gt(self,index):

        img_path = self.img_path[index]
        # print(img_path)

        mask_path = self.mask_path[index]

        img = Image.open(img_path)
        if img.mode is not 'RGB':
            img=img.convert('RGB')
        w, h = img.size
        mask_map =  Image.open(mask_path)  
        return img, mask_map

    def read_box_gt(self,box_gt_file):
        gt_data = {}
        with open(box_gt_file) as f:
            for line in f.readlines():
                line = line.strip().split(' ')

                line_data = [int(i) for i in line]
                idx, num = [line_data[0], line_data[1]]
                points_r = []
                if num > 0:
                    points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                    gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
                else:
                    gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}

        return gt_data

    def get_num_samples(self):
        return self.num_samples


    def gen_scale_map(self,path, w, h):
        with open (path) as f:
            ImgInfo = json.load(f)
            x, y = [], []
            scale_map = np.zeros((h, w))
            for head in ImgInfo["points"]:
                x.append(int(head[0]))
                y.append(int(head[1]))

            for (w_start, h_start, w_end, h_end) in ImgInfo["boxes"]:
                w_start, h_start, w_end, h_end = int(w_start), int(h_start), int(w_end), int(h_end)
                area = (w_end - w_start) * (h_end - h_start)
                area = area if area>1 else 1
                size = np.log(area)
                scale_map[h_start:h_end, w_start:w_end] = size
            return (Image.fromarray(scale_map))

    def gen_mask_map(self,path, w, h):
        with open (path) as f:
            ImgInfo = json.load(f)
            x, y = [], []
            mask_map = np.zeros((h, w))
            for head in ImgInfo["points"]:
                x.append(int(head[0]))
                y.append(int(head[1]))

            for (w_start, h_start, w_end, h_end) in ImgInfo["boxes"]:

                w_start, h_start, w_end, h_end = int(w_start), int(h_start), int(w_end), int(h_end)
                # print(w_start, h_start, w_end, h_end)
                center_w = int(np.round((w_end+w_start)/2))
                center_h = int(np.round((h_end + h_start) / 2))
                width= max(int(np.round((w_end-w_start)/2)),1)
                height = max(int(np.round((h_end-h_start)/2)),1)
                assert  (width>0 and height>0)
                # print(width, height)
                if width>height:
                    if width>7:
                        height = math.ceil(height * (7/width))
                        width  = 7
                else:
                    if height>7:
                        width  = math.ceil(width * (7/height))
                        height  = 7
                # print(width, height,center_w + width,w)
                height = height if (center_h+height)<h-1 else  h-1-center_h
                width = width if (center_w + width) <w-1 else  w-1-center_w

                height = height if (center_h-height)>0 else 0
                width = width if (center_w - width) >0 else 0

                mask = self.generate_mask(height, width)

                mask_map[center_h-height:center_h+height+1,center_w-width:center_w+width+1] += mask

                # maks_map[h_start:h_end, w_start:w_end] = mask
            mask_map[mask_map>1] =0
            return (Image.fromarray(mask_map))

    def generate_mask(self,height, width):
        y, x = np.ogrid[-height:height + 1, -width:width + 1]
        # ellipse mask
        mask = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
        return mask.astype(int)

if __name__ == "__main__":
    def generate_mask(height, width):
        y, x = np.ogrid[-height:height, -width:width]
        # circle mask
        mask = (x ) ** 2 / (height**2) + (y ) ** 2/(width**2) <= 1
        return mask.astype(float)

    print(generate_mask(3,4))
