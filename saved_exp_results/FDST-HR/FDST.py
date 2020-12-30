from easydict import EasyDict as edict


# init
__C_FDST = edict()

cfg_data = __C_FDST

__C_FDST.TRAIN_SIZE = (512,1024)
__C_FDST.DATA_PATH = '../ProcessedData/FDST/'
__C_FDST.TRAIN_LST = 'train.txt'
__C_FDST.VAL_LST =  'val.txt'
__C_FDST.VAL4EVAL = 'val_gt_loc.txt'

__C_FDST.MEAN_STD = (
    [0.452016860247, 0.447249650955, 0.431981861591],
    [0.23242045939, 0.224925786257, 0.221840232611]
)

__C_FDST.LABEL_FACTOR = 1
__C_FDST.LOG_PARA = 1.

__C_FDST.RESUME_MODEL = ''#model path
__C_FDST.TRAIN_BATCH_SIZE = 6 #imgs

__C_FDST.VAL_BATCH_SIZE = 1 # must be 1


