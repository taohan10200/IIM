from easydict import EasyDict as edict


# init
__C_JHU = edict()

cfg_data = __C_JHU

__C_JHU.TRAIN_SIZE = (512, 1024)
__C_JHU.DATA_PATH = '../ProcessedData/JHU'
__C_JHU.TRAIN_LST = 'train.txt'
__C_JHU.VAL_LST =  'val.txt'
__C_JHU.VAL4EVAL = 'val_gt_loc.txt'

__C_JHU.MEAN_STD = ([0.42968395352363586, 0.4371049106121063, 0.4219788610935211], [0.23554939031600952, 0.2325684279203415, 0.23559504747390747])

__C_JHU.LABEL_FACTOR = 1
__C_JHU.LOG_PARA =1.
__C_JHU.RESUME_MODEL = ''#model path
__C_JHU.TRAIN_BATCH_SIZE = 6 #imgs  

__C_JHU.VAL_BATCH_SIZE = 1 # must be 1
