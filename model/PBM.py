import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class BinarizedF(Function):
  @staticmethod
  def forward(ctx, input, threshold):
    ctx.save_for_backward(input,threshold)
    a = torch.ones_like(input).cuda()
    b = torch.zeros_like(input).cuda()
    output = torch.where(input>=threshold,a,b)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    # print('grad_output',grad_output)
    input,threshold = ctx.saved_tensors
    grad_input = grad_weight  = None

    if ctx.needs_input_grad[0]:
      grad_input= 0.2*grad_output
    if ctx.needs_input_grad[1]:
      grad_weight = -grad_output
    return grad_input, grad_weight


class compressedSigmoid(nn.Module):
    def __init__(self, para=2.0, bias=0.2):
        super(compressedSigmoid, self).__init__()

        self.para = para
        self.bias = bias

    def forward(self, x):
        output = 1. / (self.para + torch.exp(-x)) + self.bias
        return output

class BinarizedModule(nn.Module):
  def __init__(self, input_channels=720):
    super(BinarizedModule, self).__init__()

    self.Threshold_Module = nn.Sequential(
        nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.PReLU(),
        # nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.PReLU(),
        # nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.PReLU(),
        nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
        nn.AvgPool2d(15, stride=1, padding=7),
    )

    self.sig = compressedSigmoid()

    self.weight = nn.Parameter(torch.Tensor(1).fill_(0.5),requires_grad=True)
    self.bias = nn.Parameter(torch.Tensor(1).fill_(0), requires_grad=True)

  def forward(self,feature, pred_map):


    p = F.interpolate(pred_map.detach(), scale_factor=0.125)
    f = F.interpolate(feature.detach(), scale_factor=0.5)
    # import pdb
    # pdb.set_trace()
    f = f * p
    threshold = self.Threshold_Module(f)

    threshold = self.sig(threshold *10.) # fixed factor

    threshold = F.interpolate(threshold, scale_factor=8)

    Binar_map = BinarizedF.apply(pred_map, threshold)
    return threshold, Binar_map



class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cnn = nn.Conv2d(1,1,3,stride=1,padding=1)

  def forward(self,input):
    output =self.cnn(input)
    return output

if __name__ == '__main__':
    from torch import optim
    import  os

    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    model = BinarizedModule().cuda()
    model0 = CNN().cuda()
    optimizer = optim.SGD(model.parameters(),lr=0.5)
    optimizer0 = optim.SGD(model0.parameters(), lr=0.05)

    loss_module =  nn.MSELoss().cuda()

    # input = Variable(torch.randn(1,3,4), requires_grad=True)
    pred = torch.Tensor([[
        [[0.5, 0.7, 0.6, 0.4],
         [0.5, 0.3, 0.3, 0.6],
         [0.2, 0.03, 0, 0.2],
         [0.3, 0.1, 0.2, 0.01]]]]).cuda()

    gt = torch.Tensor([[
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]]]).cuda()
    input = Variable(pred, requires_grad=True).cuda()

    print('input',input)
    print('gt',gt)
    for i in range(300):
        threshold = model0(input)
        print(f'threshold:{threshold.data} weight:{model.weight.data.item()} bias:{model.bias.data.item()}')

        output= model(pred,threshold)
        print('output',output)

        loss = loss_module(output,gt)
        # print('loss',loss)
        optimizer.zero_grad()
        optimizer0.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer0.step()
        print(f'loss:{loss.item()}')
        print('==' * 50)
    print(model.weight)