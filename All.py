# ''''''''''''''''''''''''''''''cfg''''''''''''''''''''''''''''''
# 
# 配置函数，包含各种训练参数的配置
# 其中，原图为 (360,480)，裁剪为 (352,480)。因为 352 可以被之后的下采样整除。
# 

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
# ''''''''''''''''''''''''''''''cfg''''''''''''''''''''''''''''''


# ''''''''''''''''''''''''''''''dataset.py''''''''''''''''''''''''''''''
# 
# 数据预处理文件，重中之中，需要手敲一遍
# 1. 标签处理
# 2. 标签编码
# 3. 可视化编码过程
# 4. 定义预处理类
# 

"""补充内容见 data process and load.ipynb"""
import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg

# 
# 标签处理函数
# 
class LabelProcessor:
    """对标签图像的编码"""

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    # 静态方法装饰器， 可以理解为定义在类中的普通函数，可以用self.<name>方式调用
    # 在静态方法内部不可以示例属性和实列对象，即不可以调用self.相关的内容
    # 使用静态方法的原因之一是程序设计的需要（简洁代码，封装功能等）
    @staticmethod
    def read_color_map(file_path):  # data process and load.ipynb: 处理标签文件中colormap的数据
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):     # data process and load.ipynb: 标签编码，返回哈希表
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')

# 
# 图片数据集处理
# return：img，label
# 
class CamvidDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)

        img, label = self.img_transform(img, label)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        label = np.array(label)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == "__main__":

    TRAIN_ROOT = './CamVid/train'
    TRAIN_LABEL = './CamVid/train_labels'
    VAL_ROOT = './CamVid/val'
    VAL_LABEL = './CamVid/val_labels'
    TEST_ROOT = './CamVid/test'
    TEST_LABEL = './CamVid/test_labels'
    crop_size = (352, 480)
    Cam_train = CamvidDataset([TRAIN_ROOT, TRAIN_LABEL], crop_size)
    Cam_val = CamvidDataset([VAL_ROOT, VAL_LABEL], crop_size)
    Cam_test = CamvidDataset([TEST_ROOT, TEST_LABEL], crop_size)

# ''''''''''''''''''''''''''''''dataset.py''''''''''''''''''''''''''''''


# ''''''''''''''''''''''''''''''train.py''''''''''''''''''''''''''''''
# 
# 训练函数
# 

# 1. 导入需要使用的包
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation
from FCN import FCN
import cfg

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

# 参数 12 表示数据集分类数
fcn = FCN(12)
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


def train(model):
    best = [0]
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        print("Eopch is [{}/{}]".format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0

        for i, sample in enumerate(train_data):
            img_data = Variable(sample["img"].to(device))
            img_label = Variable(sample["label"].to(device))

            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metric["mean_class_accuracy"]
            train_miou += eval_metric["miou"]
            train_class_acc += eval_metric["class_accuracy"]

            print('|batch[{}/{}]|batch_loss {:.8f}|'.format(i+1, len(train_data), loss.item()))
            
    t.save(net.state_dict(), 'xxx.pth')


train(fcn)
# ''''''''''''''''''''''''''''''train.py''''''''''''''''''''''''''''''


