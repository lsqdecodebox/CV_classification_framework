# coding = utf-8
# @Time    : 19-4-30 下午22:17
# @Author  : 郭冰洋
# @File    : expression_dataset.py
# @Desc    : 若没有生成csv的标注文件，则根据文件夹进行数据集处理
import sys
# 调用文件夹下的子程序
sys.path.append('/home/by/graduation_project/classification/dataset')
sys.path.append('/home/by/graduation_project/classification/utils')
sys.path.append('/home/by/graduation_project/classification/models')

import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from data_aug import *
import cv2
import numpy as np
import random
import glob
import pandas as pd

"""
# 字母标记名称
defect_label_order = ['blowhole',  'break', 'crack','fray',
                      'free','uneven']
# 与字母标记一一对应
defect_code = {
    'Blowhole':  'blowhole',
    'Break':  'break',
    'Crack':  'crack',
    'Fray':  'fray',
    'Free':  'free',
    'Uneven':  'uneven'
}
# 与数字标记一一对应
defect_label = {
    'Blowhole':  '0',
    'Break':  '1',
    'Crack':  '2',
    'Fray':  '3',
    'Free':  '4',
    'Uneven':  '5'
}
# 用字典存储缺陷和数字标记
label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))
# 获取图片路径
def get_image_pd(img_root):
    # 利用glob指令获取图片列表（/*的个数根据文件构成确定）
    img_list = glob.glob(img_root + "/*/*.jpg")
    # 利用DataFrame指令构建图片列表的字典，即图片列表的序号与其路径一一对应
    image_pd = pd.DataFrame(img_list, columns=["ImageName"])
    # 获取文件夹名称，也可以认为是标签名称
    image_pd["label_name"]=image_pd["ImageName"].apply(lambda x:x.split("/")[-2])
    # 将标签名称转化为数字标记
    image_pd["label"]=image_pd["label_name"].apply(lambda x:defect_label[x])
    print(image_pd["label"].value_counts())
    return image_pd
"""
'''
# 缺陷字母标记名称
defect_label_order = ['norm', 'defect1', 'defect2', 'defect3', 'defect4', 'defect5', 'defect6', 'defect7',
                      'defect8', 'defect9', 'defect10', 'defect11']
# 缺陷与字母标记一一对应
defect_code = {
    '正常':    'norm',
    '不导电':  'defect1',
    '擦花':    'defect2',
    '横条压凹': 'defect3',
    '桔皮': 'defect4',
    '漏底':    'defect5',
    '碰伤':   'defect6',
    '起坑':   'defect7',
    '凸粉': 'defect8',
    '涂层开裂': 'defect9',
    '脏点': 'defect10',
    '其他':   'defect11'
}
# 缺陷与数字标记一一对应
defect_label = {
    '正常':    '0',
    '不导电':  '1',
    '擦花':    '2',
    '横条压凹': '3',
    '桔皮': '4',
    '漏底': '5',
    '碰伤': '6',
    '起坑': '7',
    '凸粉': '8',
    '涂层开裂':'9',
    '脏点': '10',
    '其他': '11'
}
# 用字典存储缺陷和数字标记
label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))
# 获取图片路径
def get_image_pd(img_root):
    img_list = glob.glob(img_root + "/*/*.jpg")
    img_list2 = glob.glob(img_root+ "/*/*/*.jpg")
    image_pd1 = pd.DataFrame(img_list, columns=["ImageName"])
    image_pd2 = pd.DataFrame(img_list2, columns=["ImageName"])
    image_pd1["label_name"]=image_pd1["ImageName"].apply(lambda x:x.split("/")[-2])
    image_pd2["label_name"]=image_pd2["ImageName"].apply(lambda x:x.split("/")[-3])
    all_pd=image_pd1.append(image_pd2)
    all_pd["label"]=all_pd["label_name"].apply(lambda x:defect_label[x])
    print(all_pd["label"].value_counts())
    return all_pd


defect_label_order = ['嗜曙红细胞', '淋巴细胞', '单核细胞', '中性白细胞']
# 与字母标记一一对应
defect_code = {
    'EOSINOPHIL':  '嗜曙红细胞',
    'LYMPHOCYTE':  '淋巴细胞',
    'MONOCYTE':  '单核细胞',
    'NEUTROPHIL':  '中性白细胞'
}
# 与数字标记一一对应
defect_label = {
    'EOSINOPHIL':  '0',
    'LYMPHOCYTE':  '1',
    'MONOCYTE':    '2',
    'NEUTROPHIL':  '3'
}

defect_label_order = ['飞机','自行车','鸟','船', '瓶子', '公车',
      '汽车', '猫', '椅子', '牛', '餐桌', '狗', '马', '摩托车',
      '人', '植物', '羊', '沙发', '火车', '电视']
# 与字母标记一一对应
defect_code = {
    'aeroplane': '飞机', 'bicycle':'自行车', 'bird':'鸟', 'boat':'船', 'bottle':'瓶子', 'bus':'公车',
      'car':'汽车', 'cat':'猫', 'chair':'椅子', 'cow':'牛', 'diningtable':'餐桌', 'dog':'狗', 'horse':'马', 'motorbike':'摩托车',
      'person':'人', 'pottedplant':'植物', 'sheep':'羊', 'sofa':'沙发', 'train':'火车', 'tvmonitor':'电视'
}
# 与数字标记一一对应
defect_label = {
    'aeroplane': '0', 'bicycle': '1', 'bird': '2', 'boat': '3', 'bottle': '4', 'bus': '5',
    'car': '6', 'cat': '7', 'chair': '8', 'cow': '9', 'diningtable': '10', 'dog': '11', 'horse': '12',
    'motorbike': '13',
    'person': '14', 'pottedplant': '15', 'sheep': '16', 'sofa': '17', 'train': '18', 'tvmonitor': '19'
}
'''
defect_label_order = ['嗜曙红细胞', '淋巴细胞', '单核细胞', '中性白细胞']
# 与字母标记一一对应
defect_code = {
    'EOSINOPHIL':  '嗜曙红细胞',
    'LYMPHOCYTE':  '淋巴细胞',
    'MONOCYTE':  '单核细胞',
    'NEUTROPHIL':  '中性白细胞'
}
# 与数字标记一一对应
defect_label = {
    'EOSINOPHIL':  '0',
    'LYMPHOCYTE':  '1',
    'MONOCYTE':    '2',
    'NEUTROPHIL':  '3'
}
# 用字典存储缺陷和数字标记
label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))
# 获取图片路径
def get_image_pd(img_root):
    # 利用glob指令获取图片列表（/*的个数根据文件构成确定）
    img_list = glob.glob(img_root + "/*/*.jpeg")
    # 利用DataFrame指令构建图片列表的字典，即图片列表的序号与其路径一一对应
    image_pd = pd.DataFrame(img_list, columns=["ImageName"])
    # 获取文件夹名称，也可以认为是标签名称
    image_pd["label_name"]=image_pd["ImageName"].apply(lambda x:x.split("/")[-2])
    # 将标签名称转化为数字标记
    image_pd["label"]=image_pd["label_name"].apply(lambda x:defect_label[x])
    print(image_pd["label"].value_counts())
    return image_pd

# 数据集
class dataset(data.Dataset):
    def __init__(self, anno_pd, transforms=None,debug=False,test=False):
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.transforms = transforms
        self.debug=debug
        self.test=test
    # 返回图片个数
    def __len__(self):
        return len(self.paths)
    # 获取每个图片
    def __getitem__(self, item):
        img_path =self.paths[item]
        img_id =img_path.split("/")[-1]
        img =cv2.imread(img_path) #BGR
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]
        if self.test:
            return torch.from_numpy(img).float(), int(label)
        else:
            return torch.from_numpy(img).float(), int(label)

# 整理图片
def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label


