from easydict import EasyDict as edict


# init
__C_SHHB = edict()

cfg_data = __C_SHHB

__C_SHHB.TRAIN_SIZE = (512,1024)
__C_SHHB.DATA_PATH = '../ProcessedData/SHHB/'
__C_SHHB.TRAIN_LST = 'train.txt'
__C_SHHB.VAL_LST =  'val.txt'
__C_SHHB.VAL4EVAL = 'val_gt_loc.txt'

__C_SHHB.MEAN_STD = (
    [0.452016860247, 0.447249650955, 0.431981861591],
    [0.23242045939, 0.224925786257, 0.221840232611]
)

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 1.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 6 #imgs

__C_SHHB.VAL_BATCH_SIZE = 1 # must be 1


