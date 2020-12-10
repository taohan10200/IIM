from model.HR_Net.seg_hrnet import get_seg_model
from model.VGG.VGG16_FPN import VGG16_FPN
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model.PBM import BinarizedModule
from torchvision import models


class Crowd_locator(nn.Module):
    def __init__(self, net_name, gpu_id, pretrained=True):
        super(Crowd_locator, self).__init__()

        if net_name == 'HR_Net':
            self.Extractor = get_seg_model()
            self.Binar = BinarizedModule(input_channels=720)
        if net_name == 'VGG16_FPN':
            self.Extractor = VGG16_FPN()
            self.Binar = BinarizedModule(input_channels=768)

        if len(gpu_id) > 1:
            self.Extractor = torch.nn.DataParallel(self.Extractor).cuda()
            self.Binar = torch.nn.DataParallel(self.Binar).cuda()
        else:
            self.Extractor = self.Extractor.cuda()
            self.Binar = self.Binar.cuda()

        self.loss_BCE = nn.BCELoss().cuda()

    @property
    def loss(self):
        return  self.head_map_loss, self.binar_map_loss

    def forward(self, img, mask_gt, mode = 'train'):
        # print(size_map_gt.max())
        feature, pre_map = self.Extractor(img)

        threshold_matrix, binar_map = self.Binar(feature,pre_map)

        if mode == 'train':
        # weight = torch.ones_like(binar_map).cuda()
        # weight[mask_gt==1] = 2
            assert pre_map.size(2) == mask_gt.size(2)
            self.binar_map_loss = (torch.abs(binar_map-mask_gt)).mean()

            self.head_map_loss = F.mse_loss(pre_map, mask_gt)

        return threshold_matrix, pre_map ,binar_map

    def test_forward(self, img):
        feature, pre_map = self.Extractor(img)

        return feature, pre_map


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import torch.nn.functional as F
    # model = Res_FPN().cuda()
    # predict = torch.load('/media/D/ht/Crowd_loc_master/exp/09-17_10-54_JHU_NWPU_Res50_SCAR_1e-05/all_ep_16_mae_3825.0_mse_8343.1_nae_12.509.pth')
    # model.load_state_dict(predict)
    # img = torch.ones(1,3,80,80).cuda()
    # gt =  torch.ones(1,1,80,80).cuda()
    # out = model(img,gt)
    # print(out)
    # input = torch.zeros(2,100)+0.0001
    # target = torch.ones(2,100)
    # loss = F.binary_cross_entropy(input,target)
    # print(loss)
    model = Res_FPN(pretrained = False).cuda()
    summary(model,(3,24,24))

    # import torch
    # import torch.nn as nn
    #
    # N, C_in, H, W, C_out = 10, 4, 16, 16, 4
    # x = torch.randn(N, C_in, H, W).float()
    # conv = nn.Sequential(
    #     nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=1, bias=False),
    #     nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=1, bias=False))
    # conv_group = nn.Sequential(
    #     nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=1, groups=4, bias=False),
    #     nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=1, groups=4, bias=False)
    # )
    #
    # y = conv(x)
    # y_group = conv_group(x)
    # conv_1x1 = nn.Conv2d(C_in, C_out, kernel_size=1)
    # print("groups=1时参数大小：%d" % sum(param.numel() for param in conv.parameters()))
    # print("groups=in_channels时参数大小：%d" % sum(param.numel() for param in conv_group.parameters()))
    # print(y_group.size())