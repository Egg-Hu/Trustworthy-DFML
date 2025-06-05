import random

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import csv
import os
import pandas as pd

# data_path='/home/hzx/fcil/dfmeta/DFL2Ldata/chest/images'
# savedir = '/home/hzx/fcil/dfmeta/DFL2Ldata/chest/split/'
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(script_dir)
data_path = os.path.join(parent_dir, 'DFL2Ldata/chest/images')
savedir = os.path.join(parent_dir, 'DFL2Ldata/chest/split/')
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']
used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
data_info = pd.read_csv('/home/hzx/fcil/dfmeta/DFL2Ldata/chest/Data_Entry_2017.csv', skiprows=[0], header=None)
# First column contains the image paths
image_name_all = [os.path.join(data_path,i) for i in np.asarray(data_info.iloc[:, 0])]
# First column contains the image paths
labels_all = np.asarray(data_info.iloc[:, 1])

image_name  = []
labels = []


for name, label in zip(image_name_all,labels_all):
    label = label.split("|")
    if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in used_labels:
        labels.append(labels_maps[label[0]])
        image_name.append(name)


path_label=list(zip(image_name,labels))
path_label.sort(key=lambda a: a[1])
print(np.array(path_label))
fo = open(savedir + 'meta_train' + ".csv", "w",newline='')
writer = csv.writer(fo)
writer.writerow(['filename','label'])
writer.writerows(np.array(path_label))
fo.close()
print("%s -OK" %'meta_train')

