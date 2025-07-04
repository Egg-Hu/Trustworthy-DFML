import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import csv

# data_path='/home/hzx/fcil/dfmeta/DFL2Ldata/aircraft/images'
# savedir = '/home/hzx/fcil/dfmeta/DFL2Ldata/aircraft/split/'
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(script_dir)
data_path = os.path.join(parent_dir, 'DFL2Ldata/aircraft/images')
savedir = os.path.join(parent_dir, 'DFL2Ldata/aircraft/split/')
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

SPLIT={
'meta_train': ["A340-300",
    "A318",
    "Falcon 2000",
    "F-16A/B",
    "F/A-18",
    "C-130",
    "MD-80",
    "BAE 146-200",
    "777-200",
    "747-400",
    "Cessna 172",
    "An-12",
    "A330-300",
    "A321",
    "Fokker 100",
    "Fokker 50",
    "DHC-1",
    "Fokker 70",
    "A340-200",
    "DC-6",
    "747-200",
    "Il-76",
    "747-300",
    "Model B200",
    "Saab 340",
    "Cessna 560",
    "Dornier 328",
    "E-195",
    "ERJ 135",
    "747-100",
    "737-600",
    "C-47",
    "DR-400",
    "ATR-72",
    "A330-200",
    "727-200",
    "737-700",
    "PA-28",
    "ERJ 145",
    "737-300",
    "767-300",
    "737-500",
    "737-200",
    "DHC-6",
    "Falcon 900",
    "DC-3",
    "Eurofighter Typhoon",
    "Challenger 600",
    "Hawk T1",
    "A380",
    "777-300",
    "E-190",
    "DHC-8-100",
    "Cessna 525",
    "Metroliner",
    "EMB-120",
    "Tu-134",
    "Embraer Legacy 600",
    "Gulfstream IV",
    "Tu-154",
    "MD-87",
    "A300B4",
    "A340-600",
    "A340-500",
    "MD-11",
    "707-320",
    "Cessna 208",
    "Global Express",
    "A319",
    "DH-82"],

    'meta_val': ["737-900",
    "757-300",
    "767-200",
    "A310",
    "A320",
    "BAE 146-300",
    "CRJ-900",
    "DC-10",
    "DC-8",
    "DC-9-30",
    "DHC-8-300",
    "Gulfstream V",
    "SR-20",
    "Tornado",
    "Yak-42"],

    'meta_test': ["737-400",
    "737-800",
    "757-200",
    "767-400",
    "ATR-42",
    "BAE-125",
    "Beechcraft 1900",
    "Boeing 717",
    "CRJ-200",
    "CRJ-700",
    "E-170",
    "L-1011",
    "MD-90",
    "Saab 2000",
    "Spitfire"],
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