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
data_path='/home/hzx/fcil/dfmeta/DFL2Ldata/cropdiseases/dataset/train'
savedir = '/home/hzx/fcil/dfmeta/DFL2Ldata/cropdiseases/split/'
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

SPLIT={
'meta_train': ['Tomato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Peach___Bacterial_spot', 'Potato___healthy', 'Apple___healthy', 'Raspberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Potato___Early_blight', 'Strawberry___Leaf_scorch', 'Tomato___Target_Spot', 'Potato___Late_blight', 'Strawberry___healthy', 'Peach___healthy', 'Tomato___Bacterial_spot', 'Blueberry___healthy', 'Grape___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Cherry_(including_sour)___healthy', 'Tomato___Leaf_Mold', 'Apple___Black_rot', 'Grape___Black_rot', 'Pepper,_bell___Bacterial_spot', 'Squash___Powdery_mildew', 'Apple___Apple_scab', 'Tomato___Early_blight', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Pepper,_bell___healthy', 'Corn_(maize)___healthy', 'Tomato___healthy', 'Apple___Cedar_apple_rust', 'Corn_(maize)___Common_rust_',
               'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Grape___Esca_(Black_Measles)', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy'],
    'meta_val': [],
    'meta_test': [],
}

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(range(0,len(folder_list)),folder_list))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.' )])
    random.shuffle(classfile_list_all[i])


for split in split_list:
    num=0
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if label_dict[i] in SPLIT[split]:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            num = num + 1
    print('split_num:',num)
    fo = open(savedir + split + ".csv", "w",newline='')
    writer = csv.writer(fo, delimiter='#')
    writer.writerow(['filename','label'])
    temp=np.array(list(zip(file_list,label_list)))
    writer.writerows(temp)
    fo.close()
    print("%s -OK" %split)