import os
import sys
import math
import numpy as np
import time
import random
import shutil
import cv2
from PIL import Image

import pdb
import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

def read_pred_and_gt(pred_file,gt_file):
    # read pred
    pred_data = {}
    with open(pred_file) as f:
        
        id_read = []
        for line in f.readlines():
            line = line.strip().split(' ')

            # check1
            if len(line) <2 or len(line) % 2 !=0 or (len(line)-2)/2 != int(line[1]):
                flagError = True
                sys.exit(1)

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            id_read.append(idx)

            points = []
            if num>0:
                points = np.array(line_data[2:]).reshape(((len(line)-2)//2,2))
                pred_data[idx] = {'num': num, 'points':points}
            else:
                pred_data[idx] = {'num': num, 'points':[]}

    # read gt
    gt_data = {}
    with open(gt_file) as f:
        for line in f.readlines():
            line = line.strip().split(' ')

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            points_r = []
            if num>0:
                points_r = np.array(line_data[2:]).reshape(((len(line)-2)//5,5))
                gt_data[idx] = {'num': num, 'points':points_r[:,0:2], 'sigma': points_r[:,2:4], 'level':points_r[:,4]}
            else:                
                gt_data[idx] = {'num': 0, 'points':[], 'sigma':[], 'level':[]}

    return pred_data, gt_data

def adjust_learning_rate(optimizer, base_lr1,base_lr2, max_iters,
        cur_iters, power=0.9):
    lr1 = base_lr1*((1-float(cur_iters)/max_iters)**(power))
    lr2 = base_lr2 * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2
    return lr1,lr2

def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )


def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)

    return writer, log_file

# Hungarian method for bipartite graph
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]
    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign

def logger_txt(log_file,epoch,scores):
    f1m_l, ap_l, ar_l, mae, mse, nae, loss= scores

    snapshot_name = 'ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)

    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('    [mae %.2f mse %.2f nae %.4f], [val loss %.4f]\n' % (mae, mse, nae, loss))
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map, binar_map,threshold_matrix, boxes):  # , flow):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    y = []

    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix)):
        if idx > 1:  # show only one group
            break

        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        threshold_color_map = cv2.applyColorMap((255 * tensor[4] / (1 + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)

        for i, box in enumerate(boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[0] + box[2], box[1] + box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))

        pil_threshold = Image.fromarray(cv2.cvtColor(threshold_color_map, cv2.COLOR_BGR2RGB))


        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')), pil_to_tensor(pil_binar.convert('RGB')),
                  pil_to_tensor(pil_threshold.convert('RGB'))])

    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch + 1), x)


def print_NWPU_summary(trainer, scores):
    f1m_l, ap_l, ar_l, mae, mse, nae, loss = scores
    train_record = trainer.train_record

    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')

        f.write('  [F1 %.4f Pre %.4f Rec %.4f ] [mae %.4f mse %.4f nae %.4f], [val loss %.4f]\n\n' % (f1m_l, ap_l, ar_l,mae, mse, nae, loss))

        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    print( '  [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f mse %.2f], [val loss %.4f]'\
            % (f1m_l, ap_l, ar_l, mae, mse, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f], [mse %.2f], [nae %.4f]' % (train_record['best_model_name'], \
                                                        train_record['best_F1'], \
                                                        train_record['best_Pre'], \
                                                        train_record['best_Rec'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_nae']) )
    print( '='*50 )  


def update_model(trainer,scores):

    F1, Pre, Rec, mae, mse, nae, loss = scores
    train_record = trainer.train_record
    log_file = trainer.log_txt
    epoch = trainer.epoch

    snapshot_name = 'ep_%d_F1_%.3f_Pre_%.3f_Rec_%.3f_mae_%.1f_mse_%.1f' % (epoch + 1, F1, Pre, Rec, mae, mse)

    if F1>train_record['best_F1']  or Pre > train_record['best_Pre'] or Rec > train_record['best_Rec'] \
        or  mae < train_record['best_mae'] or mse < train_record['best_mse'] or nae < train_record['best_nae']:

        train_record['best_model_name'] = snapshot_name
        if log_file is not None:
            logger_txt(log_file,epoch,scores)

        to_saved_weight = trainer.net.state_dict()
        torch.save(to_saved_weight, os.path.join(trainer.exp_path, trainer.exp_name, snapshot_name + '.pth'))

    if F1 > train_record['best_F1']:
        train_record['best_F1'] = F1
    if Pre > train_record['best_Pre']:
        train_record['best_Pre'] = Pre
    if Rec > train_record['best_Rec']:
        train_record['best_Rec'] = Rec

    if mae < train_record['best_mae']:           
        train_record['best_mae'] = mae
    if mse < train_record['best_mse']:
        train_record['best_mse'] = mse 
    if nae < train_record['best_nae']:
        train_record['best_nae'] = nae 

    latest_state = {'train_record':train_record, 'net':trainer.net.state_dict(), 'optimizer':trainer.optimizer.state_dict(),\
                     'scheduler':trainer.scheduler.state_dict(),'epoch': trainer.epoch, 'i_tb':trainer.i_tb, 'num_iters':trainer.num_iters,'exp_path':trainer.exp_path, \
                    'exp_name':trainer.exp_name}

    torch.save(latest_state,os.path.join(trainer.exp_path, trainer.exp_name, 'latest_state.pth'))

    return train_record


def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)


    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val




class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
