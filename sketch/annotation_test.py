import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes
from utils.tools import *

isTrain = True
isLwir = True

root_path = getRootPath()
classes_path = root_path + '/' + 'model_data/kaist_classes.txt'

Kaist_path = 'D:/BaiduNetdiskDownload/ReKaist'
annotation_dir_name = 'kaist_wash_annotation_train' if isTrain else 'kaist_wash_annotation_test'
img_dir_name = 'kaist_wash_picture_train' if isTrain else 'kaist_wash_picture_test'
img_subdir_name = 'lwir' if isLwir else 'visible'
classes, _ = get_classes(classes_path)
nums = np.zeros(len(classes))


# -----------------------------------------------------------------------------------------------------------
def convert_annotation(img_id, filelist, annotation_dirname):
    in_file = open(os.path.join(Kaist_path, '%s/%s.xml' % (annotation_dirname, img_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()  # 将一个xml文件内容转成tree

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        filelist.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


# ----------------------------------------------------------------------------------------------------------

'''列出某个文件夹下所有文件'''
imgs_path = Kaist_path + '/' + img_dir_name + '/' + img_subdir_name
print('imgs_path: ', imgs_path)


kaist_train_path = os.path.join(root_path, 'kaist_train.txt')
kaist_train = open(kaist_train_path, 'w', encoding='utf-8')

for dirpath, dirnames, filenames in os.walk(imgs_path):
    for xml_name in filenames:
        img_name = xml_name.split('.')[0]
        img_path = imgs_path + '/' + img_name + '.jpg'
        print(img_path)
        kaist_train.write(img_path)
        convert_annotation(img_name, kaist_train, annotation_dirname=annotation_dir_name)
        kaist_train.write('\n')

num_train = len(filenames)
kaist_train.close()


isTrain = False
isLwir = True

root_path = getRootPath()
classes_path = root_path + '/' + 'model_data/kaist_classes.txt'

Kaist_path = 'D:/BaiduNetdiskDownload/ReKaist'
annotation_dir_name = 'kaist_wash_annotation_train' if isTrain else 'kaist_wash_annotation_test'
img_dir_name = 'kaist_wash_picture_train' if isTrain else 'kaist_wash_picture_test'
img_subdir_name = 'lwir' if isLwir else 'visible'

imgs_path = Kaist_path + '/' + img_dir_name + '/' + img_subdir_name
print('imgs_path: ', imgs_path)


kaist_val_path = os.path.join(root_path, 'kaist_val.txt')
kaist_val = open(kaist_val_path, 'w', encoding='utf-8')

for dirpath, dirnames, filenames in os.walk(imgs_path):
    for idx, xml_name in enumerate(filenames):
        num_val = int(num_train * 0.9)

        img_name = xml_name.split('.')[0]
        img_path = imgs_path + '/' + img_name + '.jpg'
        print(img_path)
        kaist_val.write(img_path)
        convert_annotation(img_name, kaist_val, annotation_dirname=annotation_dir_name)
        kaist_val.write('\n')

        if idx >= num_val:
            break

kaist_val.close()





