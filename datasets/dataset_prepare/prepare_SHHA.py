from PIL import  Image
import os
import cv2 as cv
import matplotlib.pyplot as plt
from pylab import plot
import numpy as np
import json
from functions import euclidean_dist,  generate_cycle_mask, average_del_min
import scipy.io as scio
import glob
import torch
import torch.nn.functional as F
mode = 'train'
Root = '/media/D/GJY/ht/ProcessedData/SHHA'
train_path =  os.path.join(Root,'train_data')
test_path =  os.path.join(Root,'test_data')


dst_imgs_path = os.path.join(Root,'images')
dst_mask_path = os.path.join(Root,'mask')
dst_json_path = os.path.join(Root,'jsons')

cycle  =False

if  not os.path.exists(dst_mask_path):
    os.makedirs(dst_mask_path)

if  not os.path.exists(dst_imgs_path):
    os.makedirs(dst_imgs_path)

if  not os.path.exists(dst_json_path):
    os.makedirs(dst_json_path)


def resize_images(src_path, shift=0, resize_factor = 2):
    file_list = glob.glob(os.path.join(src_path,'images','*.jpg'))
    print(len(file_list))
    for idx, img_path in enumerate(file_list):
        img_id = img_path.split('/')[-1].split('.')[0]
        img_id =  int(img_id.split('_')[1]) + shift
        img_id = str(img_id).zfill(4)
        dst_img_path = os.path.join(dst_imgs_path, img_id+'.jpg')
        if os.path.exists(dst_img_path):
            continue
        else:
            img_ori = Image.open(img_path)
            w, h = img_ori.size
            new_w, new_h = w*2, h*2
            p_w, p_h = new_w / 1024, new_h / 768
            if p_w < 1 or p_h < 1:
                if p_w > p_h:
                    new_h = 768
                    new_w = int(new_w / p_h)
                    new_w = (new_w // 16 + 1) * 16
                else:
                    new_h = int(new_h / p_w)
                    new_h = (new_h // 16 + 1) * 16
                    new_w = 1024
            else:
                new_w, new_h = (new_w // 16) * 16, (new_h // 16) * 16
            print(img_id)
            print(w, h, new_w, new_h)
            new_img = img_ori.resize((new_w,new_h),Image.BILINEAR)
            new_img.save(dst_img_path, quality=95)



def writer_jsons():

    for idx, img_name in enumerate(os.listdir(dst_imgs_path)):
        ImgInfo = {}
        ImgInfo.update({"img_id":img_name})

        img_id = img_name.split('.')[0]

        dst_json_name = os.path.join(dst_json_path, img_id + '.json')
        if os.path.exists(dst_json_name):
            continue
        else:
            imgPath = os.path.join(dst_imgs_path, img_name)
            img = Image.open(imgPath)
            size_map = cv.imread(imgPath.replace('images','size_map'),cv.IMREAD_GRAYSCALE)
            size_map =torch.from_numpy(size_map)
            size_map = F.max_pool2d(size_map[None,None,:,:].float(), (199,199),16,99)
            size_map = F.interpolate(size_map, scale_factor=16).squeeze()
            print(size_map.size())
            size_map = size_map.numpy()

            w, h = img.size
            print(img_id)

            if img_id<='0300':
                img_id = str(int(img_id))

                gt_path = os.path.join(train_path,'ground_truth', 'GT_IMG_'+ img_id + '.mat')
                ori_imgPath =  os.path.join(train_path,'images','IMG_'+ img_id + '.jpg')
            else:
                gt_path = os.path.join(test_path, 'ground_truth', 'GT_IMG_' + str(int(img_id)-300) + '.mat')
                ori_imgPath = os.path.join(test_path, 'images', 'IMG_' +str(int(img_id)-300)+ '.jpg')
            gtInf = scio.loadmat( gt_path)  # format [ w, h ]
            # print(gtInf)
            ori_img = Image.open(ori_imgPath)
            ori_w, ori_h = ori_img.size
            print('ori', ori_w, ori_h)
            print('resize',w,h)


            w_rate, h_rate= w/ori_w, h/ori_h
            annPoints = gtInf['image_info'][0,0][0,0][0]

            # print(annPoints)
            annPoints[:, 0]=  annPoints[:, 0] * w_rate
            annPoints[:, 1] = annPoints[:, 1] * h_rate
            annPoints = annPoints.astype(int)

            ImgInfo.update({"human_num": len(annPoints)})
            center_w, center_h = [], []
            xy=[]
            wide,heiht = [],[]
            for head in annPoints:

                x, y = min(head[0], w-1), min(head[1], h-1)
                center_w.append(x)
                center_h.append(y)
                xy.append([int(head[0]),int(head[1])])

                if ImgInfo["human_num"] > 4:
                    dists = euclidean_dist(head[None, :], annPoints)
                    dists = dists.squeeze()
                    id = np.argsort(dists)
                    p1_y, p1_x = min(annPoints[id[1]][1], h-  1), min(annPoints[id[1]][0], w - 1)
                    p2_y, p2_x = min(annPoints[id[2]][1], h - 1), min(annPoints[id[2]][0], w - 1)
                    p3_y, p3_x = min(annPoints[id[3]][1], h - 1), min(annPoints[id[3]][0], w - 1)
                    # print(id)
                    # import pdb
                    scale = average_del_min([size_map[y,x], size_map[p1_y, p1_x], size_map[p2_y, p2_x], size_map[p3_y, p3_x]])

                    scale = max(scale,4)
                else:
                    scale = max(size_map[y, x], 4)
                # print(x,y, scale)
                area= np.exp(scale)
                length  = int(np.sqrt(area))
                wide.append(length)
                heiht.append(length)
            ImgInfo.update({"points": xy})

            xywh=[]
            for _,(x, y, x_len, y_len) in enumerate(zip(center_w,center_h,wide,heiht)):
                # print(x,y,x_len,y_len)

                x_left_top, y_left_top         = max( int( x - x_len / 2) , 0),   max( int(y - y_len / 2) , 0)
                x_right_bottom, y_right_bottom = min( int(x +  x_len/ 2), w-1),  min( int(y+  y_len / 2), h-1)
                xywh.append([x_left_top,y_left_top,x_right_bottom,y_right_bottom])

            ImgInfo.update({"boxes": xywh})
            # print(ImgInfo)

            # plot(center_w, center_h, 'g*')
            # plt.imshow(img)
            # for (x_, y_, w_, h_) in ImgInfo["boxes"]:
            #     plt.gca().add_patch(plt.Rectangle((x_, y_), w_ - x_, h_ - y_, fill=False, edgecolor='r', linewidth=1))
            # plt.show()

            with open(dst_json_name, 'w') as  f:
                json.dump(ImgInfo, f)


def generate_masks():
    file_list = glob.glob(os.path.join(dst_imgs_path,'*.jpg'))

    print(len(file_list))
    for idx, img_path in enumerate(file_list):
        if '.jpg' in img_path :

            img_id = img_path.split('/')[-1].split('.')[0]
            img_ori = Image.open(img_path)
            w, h = img_ori.size

            print(img_id)
            print(w, h)
            mask_map = np.zeros((h, w), dtype='uint8')
            gt_name = os.path.join(dst_json_path, img_id.split('.')[0] + '.json')

            with open(gt_name) as f:
                ImgInfo = json.load(f)

            centroid_list = []
            wh_list = []
            for id,(w_start, h_start, w_end, h_end) in enumerate(ImgInfo["boxes"],0):
                centroid_list.append([(w_end + w_start) / 2, (h_end + h_start) / 2])
                wh_list.append([max((w_end - w_start) / 2, 3), max((h_end - h_start) / 2, 3)])
            # print(len(centroid_list))
            centroids = np.array(centroid_list.copy(),dtype='int')
            wh        = np.array(wh_list.copy(),dtype='int')
            wh[wh>25] = 25
            human_num = ImgInfo["human_num"]
            for point in centroids:
                point = point[None,:]

                dists = euclidean_dist(point, centroids)
                dists = dists.squeeze()
                id = np.argsort(dists)

                for start, first  in enumerate(id,0):
                    if  start>0 and start<5:
                        src_point = point.squeeze()
                        dst_point = centroids[first]

                        src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                        dst_w, dst_h = wh[first][0], wh[first][1]

                        count = 0
                        threshold_w, threshold_h = max(-int(max(src_w,dst_w)/2.),-60), max(-int(max(src_h,dst_h)/2.),-60)
                        # threshold_w, threshold_h = -5,-5
                        while  (src_w+ dst_w)-np.abs(src_point[0]-dst_point[0])>threshold_w and  (src_h+ dst_h)-np.abs(src_point[1]-dst_point[1])>threshold_h:

                            if (dst_w * dst_h) > (src_w * src_h):
                                wh[first][0] = max(int(wh[first][0] * 0.9), 1)
                                wh[first][1] = max(int(wh[first][1] * 0.9), 1)
                                dst_w, dst_h = wh[first][0], wh[first][1]
                            else:
                                wh[id[0]][0] = max(int(wh[id[0]][0]*0.9), 1)
                                wh[id[0]][1] = max(int(wh[id[0]][1]*0.9), 1)
                                src_w, src_h = wh[id[0]][0], wh[id[0]][1]


                            if human_num >=3:
                                dst_point_ = centroids[id[start+1]]
                                dst_w_, dst_h_ = wh[id[start+1]][0], wh[id[start+1]][1]
                                if (dst_w_*dst_h_) > (src_w*src_h) and (dst_w_*dst_h_) > (dst_w*dst_h):
                                    if (src_w+ dst_w_)-np.abs(src_point[0]-dst_point_[0])>threshold_w and  (src_h+ dst_h_)-np.abs(src_point[1]-dst_point_[1])>threshold_h:

                                        wh[id[start+1]][0] = max(int(wh[id[start+1]][0] * 0.9), 1)
                                        wh[id[start+1]][1] = max(int(wh[id[start+1]][1] * 0.9), 1)


                            count+=1
                            if count>50:
                                break
            for (center_w, center_h), (width, height)  in zip (centroids, wh):
                assert (width > 0 and height > 0)

                if (0 < center_w < w) and (0 < center_h < h):
                    h_start = (center_h - height)
                    h_end   = (center_h + height)

                    w_start = center_w - width
                    w_end   = center_w + width
                    #
                    if h_start <0:
                        h_start = 0

                    if h_end >h:
                        h_end = h

                    if w_start<0:
                        w_start =0

                    if w_end >w:
                        w_end = w

                    if cycle:
                        mask = generate_cycle_mask(height,width)
                        mask_map[h_start:h_end, w_start: w_end] = mask

                    else:
                        mask_map[h_start:h_end, w_start: w_end] = 1

            mask_map = mask_map*255
            cv.imwrite(os.path.join(dst_mask_path, img_id+'.png'), mask_map, [cv.IMWRITE_PNG_BILEVEL, 1])

            # plt.imshow(img_ori)
            #
            # saveImg = plt.gca()
            # plt.imshow(img_ori)
            # for a, b in zip(centroid_list, wh_list):
            #     x_, y_, w_, h_ = a[0], a[1], b[0], b[1]
            #     saveImg.add_patch(plt.Rectangle((x_ - w_, y_ - h_), 2 * w_, 2 * h_, fill=False, edgecolor='g', linewidth=1))
            #
            # saveImg.axes.get_yaxis().set_visible(False)
            # saveImg.axes.get_xaxis().set_visible(False)
            # saveImg.spines['top'].set_visible(False)
            # saveImg.spines['bottom'].set_visible(False)
            # saveImg.spines['left'].set_visible(False)
            # saveImg.spines['right'].set_visible(False)
            # dst_vis_path = os.path.join(Root, 'box_vis')
            # if not os.path.exists(dst_vis_path):
            #     os.makedirs(dst_vis_path)
            # plt.savefig(os.path.join(dst_vis_path, img_id + '.jpg'),
            #             bbox_inches='tight', pad_inches=0, dpi=300)
            # plt.close()

            # for a, b in zip(centroids,wh):
            #     x_, y_, w_, h_ = a[0], a[1], b[0], b[1]
            #
            #     plt.gca().add_patch(plt.Rectangle((x_-w_, y_-h_), 2*w_ , 2*h_, fill=False, edgecolor='r', linewidth=1))
            # plt.show()
            # plt.imshow(img_ori)



        # plot(x, y, 'r*')
        # plt.imshow(img_ori)
        # for (x_, y_, w_, h_) in ImgInfo["boxes"]:
        #     plt.gca().add_patch(plt.Rectangle((x_, y_), w_ - x_, h_ - y_, fill=False, edgecolor='r', linewidth=1))
        # plt.show()

        # print(mask_map.min(),mask_map.max())
        # mask_map = Image.fromarray(mask_map).convert('L')
        # plt.imshow(mask_map)
        # for (x_, y_, w_, h_) in ImgInfo["boxes"]:
        #     plt.gca().add_patch(plt.Rectangle((x_, y_), w_ - x_, h_ - y_, fill=False, edgecolor='r', linewidth=1))
        # plt.show()


        # delet_map = Image.fromarray(delet_map).convert('L')
        # plt.imshow(delet_map)
        # for (x_, y_, w_, h_) in ImgInfo["boxes"]:
        #     plt.gca().add_patch(plt.Rectangle((x_, y_), w_ - x_, h_ - y_, fill=False, edgecolor='r', linewidth=1))
        # plt.show()

def divide_dataset(val_ration =0.1):
    import random
    all_file = os.listdir(dst_imgs_path)
    test_set = []
    train_val= []
    for img_name in all_file:
        img_id = img_name.split('.')[0]
        if img_id>'0300':
            test_set.append(img_id)
        else:
            train_val.append(img_id)
    print("test_set_num:", len(test_set), 'train_val_num:',len(train_val))

    val_set = random.sample(train_val, round(val_ration * len(train_val)))
    print("val_set_num:", len(val_set))
    train_val = set(train_val)
    val_set   = set(val_set)
    train_set = train_val - val_set
    print("train_set_num:", len(train_set))

    train_set = sorted(train_set)
    val_set = sorted(val_set)
    test_set = sorted(test_set)
    print(test_set)
    with open(os.path.join(Root,'train.txt'), "w") as f:
        for train_name in train_set:
            f.write(train_name+'\n')
    f.close()

    with open(os.path.join(Root,'val.txt'), "w") as f:
        for valid_name in val_set:
            f.write(valid_name+'\n')

    f.close()

    with open(os.path.join(Root,'test.txt'), "w") as f:
        for valid_name in test_set:
            f.write(valid_name+'\n')

    f.close()

def loc_gt_make(  mode = 'test'):
    txt_path = os.path.join(Root,mode+'.txt')
    with open(txt_path) as f:
        lines = f.readlines()
    img_ids = []
    for line in lines:
        img_ids.append(line.split('\n')[0])


    count = 0
    for idx, img_id in enumerate(img_ids):
        print(img_id)
        json_path = os.path.join(dst_json_path, img_id+'.json')
        Box_Info = []
        Box_Info.append(img_id)
        if idx != -1:

            with open(json_path) as f:
                infor = json.load(f)

            Box_Info.append(str(infor['human_num']))
            for id, head in enumerate(infor['boxes']):
                x1, y1, x2, y2 = int(head[0]), int(head[1]), int(head[2]), int(head[3])
                center_x, center_y, w, h = int((x1+x2)/2), int((y1+y2)/2),  int((x2-x1)),int((y2-y1)),
                area = w * h
                if area == 0:
                    count += 1
                    continue

                level_area = 0
                if area >= 1 and area < 10:
                    level_area = 0
                elif area > 10 and area < 100:
                    level_area = 1
                elif area > 100 and area < 1000:
                    level_area = 2
                elif area > 1000 and area < 10000:
                    level_area = 3
                elif area > 10000 and area < 100000:
                    level_area = 4
                elif area > 100000:
                    level_area = 5

                r_small = int(min(w, h) / 2)
                r_large = int(np.sqrt (w * w + h * h) / 2)

                Box_Info.append(str(center_x))
                Box_Info.append(str(center_y))
                Box_Info.append(str(r_small))
                Box_Info.append(str(r_large))
                Box_Info.append(str(level_area))

            # print(Box_Info)
            with open(os.path.join(Root,  mode + '_gt_loc.txt'), 'a') as f:
                for ind, num in enumerate(Box_Info, 1):
                    if ind < len(Box_Info):
                        f.write(num + ' ')
                    else:
                        f.write(num)
                f.write('\n')

    print(count)
if __name__ == '__main__':
    #================1. resize images ===================
    resize_images(train_path, 0)
    resize_images(test_path, 300)

    # ================2. size_map ==================
    from datasets.dataset_prepare.scale_map import main
    main ('SHHA')

    # ================3. box_level annotations ==================
    writer_jsons()

    # ================4. masks ==================
    generate_masks()

    # ================5. train test val id==================
    divide_dataset()

    # ==============6. generate val_loc_gt.txt and test_loc_gt.txt==================
    loc_gt_make(mode = 'test')
    loc_gt_make(mode='val')

    print("task is finished")