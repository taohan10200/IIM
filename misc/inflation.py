# -*- coding: utf-8 -*-

import torch
import pdb
import torch. nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy

class inflation(nn.Module):
	def __init__(self,K=15,stride=1,padding=None):
		super(inflation,self).__init__()
		weight = numpy.zeros((K,K))
		t = (K-1)/2
		for i in range(K):
			for j in range(K):
				if abs(i-t)+abs(j-t)<=t:
					weight[i,j] = 1
		if padding is None:
			padding = K//2
		self.ikernel = nn.Conv2d(1,1,K,stride=stride,padding=padding,bias=False)
		self.ikernel.weight = torch.nn.Parameter(torch.from_numpy(weight.reshape(1,1,K,K).astype(numpy.float32)))
		for para in self.parameters():
			para.requires_grad = False

	def forward(self,x):
		x = x.unsqueeze(0)
		x = x.unsqueeze(0)
		x = self.ikernel(x)
		return x.squeeze()

class Expend(torch.nn.Module):
    def __init__(self):
        super(Expend, self).__init__()
        self.ex = torch.nn.AvgPool2d(15,stride=1,padding=7)
        for para in self.parameters():
            para.requires_grad = False

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.ex(x)
        return x.squeeze()