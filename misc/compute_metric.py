import numpy as np
from scipy import spatial as ss

from .utils import hungarian,AverageMeter,AverageCategoryMeter

num_classes = 6

def compute_metrics(dist_matrix,match_matrix,pred_num,gt_num,sigma,level):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]
        match_matrix[i_pred_p,:] = pred_dist<=sigma
        
    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]
    level_list = level[tp_gt_index]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index]==i_class).sum()
        fn_c[i_class] = (level[fn_gt_index]==i_class).sum()

    return tp,fp,fn,tp_c,fn_c



def eval_metrics(num_classes, pred_data, gt_data_T):
    # print(gt_data_T)
    if gt_data_T['num']>0:
        gt_data = {'num':gt_data_T['num'].numpy().squeeze(), 'points':gt_data_T['points'].numpy().squeeze(),\
                   'sigma':gt_data_T['sigma'].numpy().squeeze(), 'level':gt_data_T['level'].numpy().squeeze()}
    else:
        gt_data = {'num':0, 'points':[],'sigma':[], 'level':[]}

    # print(gt_data)
    tp_s,fp_s,fn_s,tp_l,fp_l,fn_l = [0,0,0,0,0,0]
    tp_c_s = np.zeros([num_classes])
    fn_c_s = np.zeros([num_classes])
    tp_c_l = np.zeros([num_classes])
    fn_c_l = np.zeros([num_classes])

    if gt_data['num'] ==0 and pred_data['num'] !=0:
        pred_p =  pred_data['points']
        fp_pred_index = np.array(range(pred_p.shape[0]))
        fp_s = fp_pred_index.shape[0]
        fp_l = fp_pred_index.shape[0]

    if pred_data['num'] ==0 and gt_data['num'] !=0:
        gt_p = gt_data['points']
        level = gt_data['level']

        fn_gt_index = np.array(range(gt_p.shape[0]))
        fn_s = fn_gt_index.shape[0]
        fn_l = fn_gt_index.shape[0]
        for i_class in range(num_classes):
            fn_c_s[i_class] = (level[fn_gt_index]==i_class).sum()
            fn_c_l[i_class] = (level[fn_gt_index]==i_class).sum()

    if gt_data['num'] !=0 and pred_data['num'] !=0:
        pred_p =  pred_data['points']
        gt_p = gt_data['points']
        sigma_s = gt_data['sigma'][:,0]
        sigma_l = gt_data['sigma'][:,1]
        level = gt_data['level']

        # dist
        dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
        match_matrix = np.zeros(dist_matrix.shape,dtype=bool)

        # sigma_s and sigma_l
        tp_s,fp_s,fn_s,tp_c_s,fn_c_s = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma_s,level)
        tp_l,fp_l,fn_l,tp_c_l,fn_c_l = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma_l,level)
    return tp_s,fp_s,fn_s,tp_c_s,fn_c_s, tp_l,fp_l,fn_l,tp_c_l,fn_c_l





if __name__ == '__main__':
    eval_metrics()
