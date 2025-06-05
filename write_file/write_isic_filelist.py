import random

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import csv
import os
# data_path='/root/dfmeta/DFL2Ldata/cifar100'
# savedir = '/root/dfmeta/DFL2Ldata/cifar100/split/'
# data_path='/data1/hzx/fcil/dfmeta/DFL2Ldata/cifar100'
# savedir = '/data1/hzx/fcil/dfmeta/DFL2Ldata/cifar100/split/'
# data_path='/home/hzx/fcil/dfmeta/DFL2Ldata/cifar100'
# savedir = '/home/hzx/fcil/dfmeta/DFL2Ldata/cifar100/split/'
import pandas as pd

data_path='/home/hzx/fcil/dfmeta/DFL2Ldata/isic/ISIC2018_Task3_Training_Input'
savedir = '/home/hzx/fcil/dfmeta/DFL2Ldata/isic/split/'
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

data_info = pd.read_csv('/home/hzx/fcil/dfmeta/DFL2Ldata/isic/ISIC2018_Task3_Training_GroundTruth.csv', skiprows=[0], header=None)
# First column contains the image paths
image_name = [os.path.join(data_path,i)+'.jpg' for i in np.asarray(data_info.iloc[:, 0])]

labels = np.asarray(data_info.iloc[:, 1:])
labels = [str(i) for i in (labels!=0).argmax(axis=1)]

for split in ['meta_train']:
    fo = open(savedir + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    path_label=list(zip(image_name,labels))
    path_label.sort(key=lambda a: a[1])
    print(np.array(path_label))
    writer.writerows(np.array(path_label))
    fo.close()
    print("%s -OK" %split)