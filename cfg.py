# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 配置函数，包含各种训练参数的配置
# 其中，原图为 (360,480)，裁剪为 (352,480)。因为 352 可以被之后的下采样整除。
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


BATCH_SIZE = 4
EPOCH_NUMBER = 2
TRAIN_ROOT = './CamVid/train'
TRAIN_LABEL = './CamVid/train_labels'
VAL_ROOT = './CamVid/val'
VAL_LABEL = './CamVid/val_labels'
TEST_ROOT = './CamVid/test'
TEST_LABEL = './CamVid/test_labels'
class_dict_path = './CamVid/class_dict.csv'
crop_size = (352, 480)
