import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
import numpy
import pdb
import cv2
from torchvision.transforms import functional as TrF
from misc import inflation

class ProcessSub(object):
    def __init__(self,T=0.1,K=51):
        self.T = T
        self.inf = inflation.inflation(K=K)

    def getHS(self,flow):
        # h direction  s or v magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h = ang * 180 / np.pi / 2 #angle
        s = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude
        return h,s

    def __call__(self, flow):
        h, s = self.getHS(flow[:, :, 0:2])
        flow[:,:, 0] = h.astype(np.float32) / 255
        flow[:,:, 1] = s.astype(np.float32) / 255
        # Threshold
        temp = np.ones(flow[:,:,2].shape)
        temp[abs(flow[:,:,2])<self.T] = 0
        flow[:,:,2] = flow[:,:,2] * temp
        # inflation
        return flow

# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img,mask = t(img, mask)
            return img,mask
        for t in self.transforms:
            img,mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                # for i in range(3):
                #     flow[:,:,i] = np.fliplr(flow[:,:,i])
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)#, flow
            w, h = img.size
            xmin = w - bbx[:,3]
            xmax = w - bbx[:,1]
            bbx[:,1] = xmin
            bbx[:,3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img,mask  #flow
        return img, mask, bbx


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None ):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask

        assert w >= tw
        assert h >= th

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # flow = flow[y1:y1+th,x1:x1+tw,:]
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)) #.flow




class ScaleByRateWithMin(object):
    def __init__(self, rateRange, min_w, min_h):
        self.rateRange = rateRange
        self.min_w = min_w
        self.min_h = min_h
    def __call__(self, img, mask):# dot, flow):
        w, h = img.size
        # print('ori',w,h)
        rate = random.uniform(self.rateRange[0], self.rateRange[1])
        new_w = int(w * rate) // 32 * 32
        new_h = int(h * rate) // 32 * 32
        if new_h< self.min_h or new_w<self.min_w:
            if new_w<self.min_w:
                new_w = self.min_w
                rate = new_w/w
                new_h = int(h*rate) // 32*32
            if new_h < self.min_h:
                new_h = self.min_h
                rate = new_h / h
                new_w =int( w * rate) //32*32

        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        return img, mask 

# ===============================image tranforms============================

class RGB2Gray(object):
    def __init__(self, ratio):
        self.ratio = ratio  # [0-1]

    def __call__(self, img):
        if random.random() < 0.1:
            return  TrF.to_grayscale(img, num_output_channels=3)
        else: 
            return img

class GammaCorrection(object):
    def __init__(self, gamma_range=[0.4,2]):
        self.gamma_range = gamma_range 

    def __call__(self, img):
        if random.random() < 0.5:
            gamma = random.uniform(self.gamma_range[0],self.gamma_range[1])
            return  TrF.adjust_gamma(img, gamma)
        else: 
            return img

# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()




# class tensormul(object):
#     def __init__(self, mu=255.0):
#         self.mu = 255.0
    
#     def __call__(self, _tensor):
#         _tensor.mul_(self.mu)
#         return _tensor
