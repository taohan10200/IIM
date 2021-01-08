from PIL import  Image
import os
import cv2 as cv
import matplotlib.pyplot as plt
from pylab import plot
import numpy as np
import json
import math
import glob
from functions import euclidean_dist,  generate_cycle_mask, average_del_min
cycle = False

Root = '/media/D/GJY/ht/ProcessedData/JHU'
dst_Root = '/media/D/GJY/ht/ProcessedData/JHU'



dst_imgs_path = os.path.join(dst_Root,'images')
dst_json_path = os.path.join(dst_Root,'jsons')
dst_mask_path = os.path.join(dst_Root,'mask')

if  not os.path.exists(dst_mask_path):
    os.makedirs(dst_mask_path)

if  not os.path.exists(dst_imgs_path):
    os.makedirs(dst_imgs_path)

if  not os.path.exists(dst_json_path):
    os.makedirs(dst_json_path)


def resize_images(mode):


    imgs_path = os.path.join(Root, mode, 'images')
    gt_path = os.path.join(Root, mode, 'gt')
    mni_size = (1024, 768)
    file_list = os.listdir(imgs_path)
    print(len(file_list))
    for imgName in file_list:
        img_id = imgName.split('.')[0]
        img_path = os.path.join(imgs_path, imgName)
        dst_img_path = os.path.join(dst_imgs_path, imgName)
        #==============================image resize================
        # if os.path.exists(dst_img_path):
        #     continue
        # else:
        img_ori = Image.open(img_path)
        ori_w, ori_h = img_ori.size
        p_w, p_h = ori_w/1024, ori_h/768
        if p_w<1 or p_h<1:
            if p_w > p_h:
                new_h = 768
                new_w = int(ori_w/p_h)
                new_w = (new_w//16+1)*16
            else:
                new_h = int(ori_h/p_w)
                new_h = (new_h // 16 + 1) * 16
                new_w = 1024
        else:
            new_w, new_h = (ori_w//16)*16, (ori_h//16)*16
        print(img_id)
        print(ori_w, ori_h, new_w, new_h)
        new_img = img_ori.resize((new_w,new_h),Image.BILINEAR)
        new_img.save(dst_img_path, quality=95)


        # ============================gt resize================
        ImgInfo = {}
        ImgInfo.update({"img_id":img_id})

        gt_name = os.path.join(gt_path, img_id+'.txt')
        infor = []
        with open(gt_name) as f:
            lines = f.readlines()
        for line in lines:
            infor.append(line.split('\n')[0])

        ImgInfo.update({"human_num": len(infor)})
        x_list, y_list = [], []
        xy_list = []
        w_list, h_list = [], []
        w_rate,h_rate = new_w/ori_w, new_h/ori_h
        for head in infor:
            splits = head.split()
            x_c, y_c, w, h, = int(splits[0])*w_rate, int(splits[1])*h_rate, \
                                        int(splits[2])*w_rate, int(splits[3])*h_rate
            x_list.append(x_c-w/2.)
            y_list.append(y_c-h/2.)
            xy_list.append([x_c, y_c])
            max_len= max(int(splits[2])*w_rate, int(splits[3])*h_rate)
            w_list.append(max_len)
            h_list.append(max_len)
        ImgInfo.update({"points": xy_list})

        xyxy = []
        for _, (x_s, y_s, w, h) in enumerate(zip(x_list, y_list, w_list, h_list)):

            x_end = x_s+w
            y_end = y_s+h

            xyxy.append([x_s, y_s, x_end, y_end])

        ImgInfo.update({"boxes": xyxy})
        # print(ImgInfo)

        with open(os.path.join(dst_json_path, img_id+ '.json'), 'w') as  f:
            json.dump(ImgInfo, f)


        # img_new = np.array(new_img)
        # plot(x, y, 'r*')
        #
        # plt.imshow(img_new)
        # for _,(x_s, y_s, x_end, y_end) in enumerate(xyxy):
        #     w_ = int(x_end - x_s )
        #     h_ = int(y_end - y_s )
        #     plt.gca().add_patch( plt.Rectangle((x_s, y_s), w_,h_, fill=False,edgecolor='r', linewidth=1))
        # plt.show()


def generate_masks():
    file_list = glob.glob(os.path.join(dst_imgs_path,'*.jpg'))

    print(len(file_list))
    for idx, img_path in enumerate(file_list):
        if idx <-1 :
            break
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
        wh[wh>15] = 15
        human_num = ImgInfo["human_num"]
        for point in centroids:
            point = point[None,:]

            dists = euclidean_dist(point, centroids)
            dists = dists.squeeze()
            id = np.argsort(dists)

            for start, first in enumerate(id, 0):
                if start > 0 and start < 5:
                    src_point = point.squeeze()
                    dst_point = centroids[first]

                    src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                    dst_w, dst_h = wh[first][0], wh[first][1]

                    count = 0
                    if (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > 0 and (src_h + dst_h) - np.abs(src_point[1] - dst_point[1]) > 0:
                        w_reduce = ((src_w + dst_w) - np.abs(src_point[0] - dst_point[0])) / 2
                        h_reduce = ((src_h + dst_h) - np.abs(src_point[1] - dst_point[1])) / 2
                        threshold_w, threshold_h = max(-int(max(src_w - w_reduce, dst_w - w_reduce) / 2.), -60), max(
                            -int(max(src_h - h_reduce, dst_h - h_reduce) / 2.), -60)

                    else:
                        threshold_w, threshold_h = max(-int(max(src_w, dst_w) / 2.), -60), max(-int(max(src_h, dst_h) / 2.), -60)
                    # threshold_w, threshold_h = -5, -5
                    while (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > threshold_w and (src_h + dst_h) - np.abs(
                            src_point[1] - dst_point[1]) > threshold_h:

                        if (dst_w * dst_h) > (src_w * src_h):
                            wh[first][0] = max(int(wh[first][0] * 0.9), 2)
                            wh[first][1] = max(int(wh[first][1] * 0.9), 2)
                            dst_w, dst_h = wh[first][0], wh[first][1]
                        else:
                            wh[id[0]][0] = max(int(wh[id[0]][0] * 0.9), 2)
                            wh[id[0]][1] = max(int(wh[id[0]][1] * 0.9), 2)
                            src_w, src_h = wh[id[0]][0], wh[id[0]][1]

                        if human_num >= 3:
                            dst_point_ = centroids[id[start + 1]]
                            dst_w_, dst_h_ = wh[id[start + 1]][0], wh[id[start + 1]][1]
                            if (dst_w_ * dst_h_) > (src_w * src_h) and (dst_w_ * dst_h_) > (dst_w * dst_h):
                                if (src_w + dst_w_) - np.abs(src_point[0] - dst_point_[0]) > -3 and (src_h + dst_h_) - np.abs(
                                        src_point[1] - dst_point_[1]) > -3:
                                    wh[id[start + 1]][0] = max(int(wh[id[start + 1]][0] * 0.9), 2)
                                    wh[id[start + 1]][1] = max(int(wh[id[start + 1]][1] * 0.9), 2)

                        count += 1
                        if count > 40:
                            break
        for (center_w, center_h), (width, height) in zip(centroids, wh):
            assert (width > 0 and height > 0)

            if (0 < center_w < w) and (0 < center_h < h):
                h_start = (center_h - height)
                h_end = (center_h + height )

                w_start = center_w - width
                w_end = center_w + width
                #
                if h_start < 0:
                    h_start = 0

                if h_end > h:
                    h_end = h

                if w_start < 0:
                    w_start = 0

                if w_end > w:
                    w_end = w

                if cycle:
                    mask = generate_cycle_mask(height, width)
                    mask_map[h_start:h_end, w_start: w_end] = mask

                else:
                    mask_map[h_start:h_end, w_start: w_end] = 1


        mask_map = mask_map*255

        cv.imwrite(os.path.join(dst_mask_path, img_id+'.png'), mask_map, [cv.IMWRITE_PNG_BILEVEL, 1])

        # plt.imshow(img_ori)

        saveImg = plt.gca()
        plt.imshow(img_ori)
        for a, b in zip(centroid_list,wh_list):

            x_, y_, w_, h_ = a[0], a[1], b[0], b[1]
            saveImg.add_patch(plt.Rectangle((x_-w_, y_-h_), 2*w_ , 2*h_, fill=False, edgecolor='g', linewidth=1))


        saveImg.axes.get_yaxis().set_visible(False)
        saveImg.axes.get_xaxis().set_visible(False)
        saveImg.spines['top'].set_visible(False)
        saveImg.spines['bottom'].set_visible(False)
        saveImg.spines['left'].set_visible(False)
        saveImg.spines['right'].set_visible(False)
        dst_vis_path = os.path.join(dst_Root,'box_vis')
        if not os.path.exists(dst_vis_path):
            os.makedirs(dst_vis_path)
        plt.savefig(os.path.join(dst_vis_path, img_id+'.jpg'),
                    bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        # plt.show()

        # for a, b in zip(centroids,wh):
        #     x_, y_, w_, h_ = a[0], a[1], b[0], b[1]
        #
        #     plt.gca().add_patch(plt.Rectangle((x_-w_, y_-h_), 2*w_ , 2*h_, fill=False, edgecolor='r', linewidth=1))
        #
        # plt.imshow(img_ori)
        # plt.show()



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
def loc_gt_make(  mode = 'test'):
    txt_path = os.path.join(dst_Root,mode+'.txt')
    with open(txt_path) as f:
        lines = f.readlines()
    img_ids = []
    for line in lines:
        img_ids.append(line.split('\n')[0])


    count = 0
    for idx, img_id in enumerate(img_ids):
        print(img_id)
        json_path = os.path.join(dst_Root, 'jsons', img_id+'.json')
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
            with open(os.path.join(dst_Root,  mode + '_gt_loc.txt'), 'a') as f:
                for ind, num in enumerate(Box_Info, 1):
                    if ind < len(Box_Info):
                        f.write(num + ' ')
                    else:
                        f.write(num)
                f.write('\n')

    print(count)

def JHU_list_make(mode):
    path = os.path.join(Root,mode,'images')
    for filname in os.listdir(path):
        filname = filname.split('.')[0]
        with open(dst_Root+'/'+mode+'.txt','a') as f:
            f.write(filname + '\n')

if __name__ == '__main__':
    #================1. resize images and gt===================
    resize_images('train')
    resize_images('val')
    resize_images('test')

    # ================2. masks==================
    generate_masks()

    # ================3. train test val id==================
    JHU_list_make('test')
    JHU_list_make('val')
    JHU_list_make('train')

    # ================4. generate val_loc_gt.txt and test_loc_gt.txt==================
    loc_gt_make(mode = 'test')
    loc_gt_make(mode='val')