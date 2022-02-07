# Dataset preparation

We provide the prepared datasets that you can directly download from this [link](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EliCeOckaZVBgez6n8ZWvr4BNdwPauFJgbm88MGhHid25w?e=rtogwc)


```[Optional]``` If you want to prepare the dataset by yourself or test our codes on your dataset, you will need to set up the dataset with the following instructions.
### Preparation
- Download the pretrained scale prediction model (```pretrained_scale_prediction_model.pth```) from the [link](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EliCeOckaZVBgez6n8ZWvr4BNdwPauFJgbm88MGhHid25w?e=rtogwc).
- Place the model on the root: ``` $IIM/datasets/dataset_prepare```
### Shanghai Tech Part A
- Download the images (train_data, test_data) from [[Link](https://1drv.ms/u/s!AgKz_E1uf260lkYiv3Midn3eU3tW?e=EWHThx)].
- Place the train_data and test_data to make the data folder like:

  ~~~
  --ProcessedData
      |-- SHHA
       -- |-- train_data
          |-- test_data
          |-- images               # To be generated
          |-- masks                # To be generated
          |-- size_map             # To be generated
          |-- jsons                # To be generated
          |-- train.txt            # To be generated
          |-- val.txt              # To be generated
          |-- test.txt             # To be generated
          |-- val_gt_loc.txt       # To be generated
          |-- test_gt_loc.txt      # To be generated
  ~~~
- To generate other folders and files, you should run the command: 

    ~~~
    cd $IIM/datasets/dataset_prepare
    python prepare_SHHA.py
    ~~~
    To generate the required files for training and testing, you should make sure the following steps are completed successfully.
    ~~~
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
        ~~~ 

### Shanghai Tech Part B
- Download the images (train_data, test_data) from [[Link](https://1drv.ms/u/s!AgKz_E1uf260lkYiv3Midn3eU3tW?e=EWHThx)].
- Place the train_data and test_data to make the data folder like the Shanghai Tech Part A dataset.

- Run 
 
    ~~~****************
    cd $IIM/datasets/dataset_prepare
    python prepare_SHHB.py
    ~~~

### UCF-QNRF
- Download the images (Train, Test) from [[Homepage](https://www.crcv.ucf.edu/data/ucf-qnrf/)] or [[Download](https://drive.google.com/open?id=1fLZdOsOXlv2muNB_bXEW6t-IS9MRziL6)].
- Place the Train and Test to make the data folder like the Shanghai Tech Part A dataset:

- Run 
 
    ~~~****************
    cd $IIM/datasets/dataset_prepare
    python prepare_QNRF.py
    ~~~


### JHU
- Download the images (train_data, test_data) from  [[Homepage](http://www.crowd-counting.com)]. 
- Place the train_data and test_data to make the data folder like.

  ~~~
  --ProcessedData
      |-- JHU
       -- |-- train
          |-- val
          |-- test
          |-- images               # To be generated
          |-- masks                # To be generated
          |-- jsons                # To be generated
          |-- train.txt            # To be generated
          |-- val.txt              # To be generated
          |-- test.txt             # To be generated
          |-- val_gt_loc.txt       # To be generated
          |-- test_gt_loc.txt      # To be generated
  ~~~
- To generate other folders and files, you should run the command: 

    ~~~
    cd $IIM/datasets/dataset_prepare
    python prepare_JHU.py
    ~~~
    To generate the required files for training and testing, make sure the following steps are completed successfully.
    ~~~
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
    
        print("task is finished")
        ~~~ 
