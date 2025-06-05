import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import csv

data_path='/home/hzx/fcil/dfmeta/DFL2Ldata/dtd/dtd/images'
savedir = '/home/hzx/fcil/dfmeta/DFL2Ldata/dtd/split/'
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

SPLIT={
'meta_train': ["chequered",
    "braided",
    "interlaced",
    "matted",
    "honeycombed",
    "marbled",
    "veined",
    "frilly",
    "zigzagged",
    "cobwebbed",
    "pitted",
    "waffled",
    "fibrous",
    "flecked",
    "grooved",
    "potholed",
    "blotchy",
    "stained",
    "crystalline",
    "dotted",
    "striped",
    "swirly",
    "meshed",
    "bubbly",
    "studded",
    "pleated",
    "lacelike",
    "polka-dotted",
    "perforated",
    "freckled",
    "smeared",
    "cracked",
    "wrinkled"],

    'meta_val': ["gauzy",
    "grid",
    "lined",
    "paisley",
    "porous",
    "scaly",
    "spiralled"],

    'meta_test': ["banded",
    "bumpy",
    "crosshatched",
    "knitted",
    "sprinkled",
    "stratified",
    "woven"],
}

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(range(0,len(folder_list)),folder_list))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


for split in split_list:
    num=0
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if label_dict[i] in SPLIT[split]:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(label_dict[i], len(classfile_list)).tolist()
            num = num + 1
    print('split_num:',num)
    fo = open(savedir + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    temp=np.array(list(zip(file_list,label_list)))
    writer.writerows(temp)
    fo.close()
    print("%s -OK" %split)