import copy
import math
import os.path
import random
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from dataset.cifar100 import Cifar100, Cifar100_Specific
from dataset.samplers import CategoriesSampler
from dataset.miniimagenet import MiniImageNet, MiniImageNet_Specific
from dataset.omniglot import Omniglot, Omniglot_Specific
from dataset.mnist import Mnist, Mnist_Specific
import network
from dataset.cub import CUB, CUB_Specific
from dataset.flower import flower_Specific, flower
from dataset.chest import chest_Specific, chest
from dataset.cropdiseases import cropdiseases_Specific, cropdiseases
from dataset.eurosat import eurosat_Specific, eurosat
from dataset.isic import isic_Specific, isic
from logger import get_logger
from network import Conv4, ResNet34, ResNet18, ResNet50, ResNet10
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def get_dataloader(args,noTransform_test=False,resolution=32):
    if args.dataset=='cifar100':
        trainset = Cifar100(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    elif args.dataset=='miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    elif args.dataset=='cub':
        trainset = CUB(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    elif args.dataset=='flower':
        trainset = flower(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    if args.testdataset == 'cifar100':
        trainset = Cifar100(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size=trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset=Cifar100(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                          600,
                                          args.way_test,
                                          args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                  num_workers=0,
                                  batch_sampler=val_sampler,
                                  pin_memory=True)
        testset = Cifar100(setname='meta_test', augment=False,noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                num_workers=8,
                                batch_sampler=test_sampler,
                                pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset == 'miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False,resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = MiniImageNet(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset == 'omniglot':
        trainset = Omniglot(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        testset = Omniglot(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        val_loader=None
        return train_loader, val_loader, test_loader
    elif args.testdataset=='cub':
        trainset = CUB(setname='meta_train', augment=False,resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = CUB(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = CUB(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset=='flower':
        trainset = flower(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        # train_sampler = CategoriesSampler(trainset.label,
        #                                   args.episode_train,
        #                                   args.way_train,
        #                                   args.num_sup_train + args.num_qur_train)
        # train_loader = DataLoader(dataset=trainset,
        #                           num_workers=8,
        #                           batch_sampler=train_sampler,
        #                           pin_memory=True)
        valset = flower(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = flower(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, val_loader, test_loader
    elif args.testdataset=='cropdiseases':
        testset = cropdiseases(setname='meta_train', augment=False, noTransform=noTransform_test)#only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, None, test_loader
    elif args.testdataset=='eurosat':
        testset = eurosat(setname='meta_train', augment=False, noTransform=noTransform_test)  # only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, None, test_loader
    elif args.testdataset=='isic':
        testset = isic(setname='meta_train', augment=False, noTransform=noTransform_test)  # only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, None, test_loader
    elif args.testdataset=='chest':
        testset = chest(setname='meta_train', augment=False, noTransform=noTransform_test)  # only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, None, test_loader
    elif args.testdataset=='mix':
        args.num_class=None
        args.img_size = None
        args.channel=None
        testset_cifar = Cifar100(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset_cifar.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_cifar = DataLoader(dataset=testset_cifar,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        testset_mini = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset_mini.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_mini = DataLoader(dataset=testset_mini,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)

        testset_cub = CUB(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset_cub.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_cub = DataLoader(dataset=testset_cub,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return test_loader_cifar, test_loader_mini, test_loader_cub
    else:
        ValueError('not implemented!')
    #return None, val_loader, test_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False

def get_model(args,type,set_maml_value=True,arbitrary_input=False):
    set_maml(set_maml_value)
    way = args.way_train
    if type == 'conv4':
        model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size,arbitrary_input=arbitrary_input,channel=args.channel)
    elif type == 'resnet34':
        model_maml = ResNet34(flatten=True, out_dim=way)
    elif type == 'resnet18':
        model_maml = ResNet18(flatten=True, out_dim=way)
    elif type == 'resnet50':
        model_maml = ResNet50(flatten=True, out_dim=way)
    elif type=='resnet10':
        model_maml = ResNet10(flatten=True, out_dim=way)
    else:
        raise NotImplementedError
    return model_maml

OOD_index={'eurosat':1,'isic':2,'chest':3,'omniglot':4,'mnist':5,'fake':6}
bias={'cifar100':0,'miniimagenet':64,'cub':128,'flower':228,'cropdiseases':299,'eurosat':337,'isic':347,'chest':354,'omniglot':361,'mnist':425}
bias_end={'cifar100':63,'miniimagenet':127,'cub':227,'flower':298,'cropdiseases':336,'eurosat':346,'isic':353,'chest':360,'omniglot':424,'mnist':434}
teacher_list=['conv4', 'resnet10', 'resnet18']
OOD_list=['eurosat', 'isic', 'chest','omniglot','mnist']
dataset_list=['cifar100', 'miniimagenet', 'cub', 'flower']
dataset_classnum={'cifar100':64,'miniimagenet':64,'cub':100,'flower':71,'cropdiseases':38,'eurosat':10,'isic':7,'chest':7,'omniglot':64,'mnist':10}


def set_maml(flag):
    network.ConvBlock.maml = flag
    network.SimpleBlock.maml = flag
    network.BottleneckBlock.maml = flag
    network.ResNet.maml = flag
    network.ConvNet.maml = flag
def set_direct(flag):
    network.Linear_fw.direct=flag
    network.Conv2d_fw.direct = flag
    network.BatchNorm2d_fw.direct = flag

def get_teacher_sequence(args):
    normal_model_num=args.APInum
    ood_model_num=int((args.oodP*normal_model_num)//(1-args.oodP))
    fake_model_num = int((args.fakeP * normal_model_num) // (1 - args.fakeP))
    print('oodP:',(ood_model_num)/(ood_model_num+normal_model_num))
    print('fakeP:', (fake_model_num) / (fake_model_num + normal_model_num))
    print('total length:',normal_model_num+ood_model_num+fake_model_num,'normal:',normal_model_num,'ood:',ood_model_num,'fake:',fake_model_num)
    ood_model_id_list=random.sample(list(range(100,600)),ood_model_num)
    fake_model_id_list = random.sample(list(range(600, 700)), fake_model_num)
    model_id_list=ood_model_id_list+fake_model_id_list+list(range(0,100))
    random.shuffle(model_id_list)

    # if args.oodP==0:
    #     model_id_list=[37, 15, 31, 23, 9, 10, 50, 67, 78, 38, 25, 44, 49, 57, 83, 22, 32, 71, 99, 54, 19, 41, 59, 77, 14, 63, 12, 13, 3, 96, 61, 11, 84, 94, 18, 42, 48, 46, 60, 73, 65,
    #                    79, 45, 0, 16, 27, 20, 89, 34, 2, 17, 28, 86, 24, 35, 64, 91, 93, 62, 76,
    #                    30, 43, 29, 6, 85, 21, 97, 75, 53, 8, 51, 98, 47, 5, 33, 4, 72, 92, 70, 58, 80, 82, 95, 55, 1, 40, 81, 87, 52, 90, 88, 66, 7, 74, 39, 69, 56, 36, 68]
    # elif args.oodP==0.2:
    #     model_id_list=[10, 53, 6, 94, 68, 58, 91, 83, 20, 18, 14, 43, 96, 99, 89, 61, 454, 26, 49, 38, 82, 4, 365, 515, 72, 0, 93, 16, 13, 35,
    #                    22, 2, 420, 7, 39, 73, 59, 320, 41, 98, 65, 19, 69, 489, 52, 50, 3, 54, 15, 21, 40, 76, 36, 1, 34, 78, 64, 74, 97, 51, 79, 461,
    #                    24, 57, 555, 106, 66, 86, 28, 450, 333, 70, 311, 87, 30, 45, 326, 90, 55, 379, 540, 60, 372, 8, 31, 17, 67, 44, 95, 11, 33, 56,
    #                    25, 47, 424, 77, 260, 63, 85, 84, 37, 62, 5, 131, 247, 430, 32, 92, 29, 501, 27, 12, 23, 88, 75, 399, 9, 71, 80, 258, 48, 42, 81, 46]
    # elif args.oodP==0.4:
    #     model_id_list=[2, 69, 71, 91, 80, 30, 17, 81, 27, 32, 24, 6, 0, 93, 307, 8,39 , 68, 258, 89, 11, 320,
    #                    64, 55, 126, 461, 18, 420, 454, 106, 12, 34, 22, 399, 38, 135, 107, 422, 10, 247, 59, 84, 98, 65, 56, 29, 599, 50, 185, 95, 60, 450, 7, 516, 40, 35, 536, 62, 154, 1, 76, 37, 13, 51, 9, 92, 387, 430, 94, 5, 379,
    #                    46, 499, 326, 70, 33, 97, 67, 47, 331, 74, 540, 20, 28, 501, 288, 87, 31, 88, 52, 23, 54,
    #                    481, 311, 548, 347, 49, 58, 21, 489, 572, 575, 119, 446, 299, 116, 77, 86, 14, 73, 57, 61, 448,
    #                    19, 375, 241, 424, 423, 15, 75, 63, 66, 82, 43, 25, 4, 444, 158, 26, 90, 96, 515, 313, 83, 85,
    #                    79, 53, 555, 120, 315, 366, 134, 99, 3, 235, 78, 48, 36, 591, 522, 217, 381, 390, 333, 260, 42, 72, 16, 41, 131, 365, 563, 44, 372, 265, 45]
    # elif args.oodP==0.6:
    #     model_id_list=[85, 15, 98, 8, 61, 20, 21, 62, 89, 76, 134, 16, 63, 79, 488, 1, 399, 19, 96, 298, 43, 261, 9, 269, 0, 415, 193,
    #                    546, 116, 119, 562, 376, 543, 461, 38, 381, 48, 529, 176, 80, 506, 454, 247, 333, 390, 34, 394, 152, 81, 447, 489, 463, 484, 466,
    #                    82, 87, 501, 101, 95, 184, 94, 595, 60, 375, 378, 581, 307, 316, 487, 84, 199, 212, 73, 151, 72, 11, 18, 7, 67, 540, 52, 555, 442,
    #                    4, 517, 313, 46, 185, 153, 427, 520, 181, 47, 107, 422, 59, 83, 57, 58, 17, 50, 36, 588, 331, 467, 321, 231, 299, 51, 491, 99,
    #                    29, 71, 35, 32, 120, 420, 209, 42, 516, 56, 336, 582, 6, 548, 150, 93, 235, 75, 387, 238, 423, 10, 136, 14, 597, 40, 97, 492,
    #                    66, 522, 165, 481, 92, 326, 285, 44, 154, 380, 27, 33, 329, 31, 167, 424, 366, 354, 337, 450, 5, 64, 565, 499, 91, 147, 12,
    #                    498, 434, 264, 68, 77, 356, 88, 53, 320, 315, 135, 174, 86, 205, 379, 365, 515, 39, 338, 536, 493, 54, 556, 446, 24, 308, 41,
    #                    444, 311, 593, 304, 289, 244, 526, 74, 288, 158, 69, 591, 567, 3, 260, 45, 505, 2, 55, 22, 28, 70, 131, 322, 168, 26,
    #                    25, 217, 258, 476, 598, 372, 347, 113, 23, 417, 448, 563, 90, 458, 49, 106, 37, 78, 480, 65, 126, 323, 428, 13, 30, 265, 241, 430, 114, 388]
    # else:
    #     raise NotImplementedError
    return model_id_list
def get_teacher_from_sequence(args,model_id_list,timestep=None):
    teacher = get_model(args=args, type=args.pre_backbone, set_maml_value=False, arbitrary_input=False).cuda(args.device)
    model_id=model_id_list[timestep]
    node_id=model_id%args.APInum
    if model_id<args.APInum:
        select_dataset=args.dataset
        pretrained_path = './pretrained/{}_{}/{}way/model'.format(select_dataset, args.pre_backbone, args.way_pretrain)
    elif model_id>=600:
        select_dataset=args.dataset
        pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(select_dataset, args.pre_backbone, args.way_pretrain)
    else:
        select_dataset=OOD_list[model_id//args.APInum-1]
        pretrained_path = './pretrained/{}_{}/{}way/model'.format(select_dataset, args.pre_backbone, args.way_pretrain)
    tmp = torch.load(os.path.join(pretrained_path, 'model_specific_acc_{}.pth'.format(node_id)))
    teacher.load_state_dict(tmp['teacher'])
    specific = tmp['specific']
    specific = [i + bias[select_dataset] for i in specific]
    if select_dataset!=args.dataset:
        specific = random.sample(range(dataset_classnum[args.dataset]), args.way_pretrain)
        specific = [i + bias[args.dataset] for i in specific]
    return teacher,specific


def get_random_teacher(args):
    if args.pre_backbone != 'mix': #ss
        teacher = get_model(args=args,type=args.pre_backbone,set_maml_value=False,arbitrary_input=False).cuda(args.device)
        node_id = random.randint(0, args.APInum - 1)
        model_id=node_id
        select_dataset=args.dataset
        pretrained_path = './pretrained/{}_{}/{}way/model'.format(args.dataset, args.pre_backbone,args.way_pretrain)
        temp = random.random()
        if temp <= args.oodP:
            if args.fakefrom == -1:
                ood_dataset = random.choice(OOD_list)
                model_id=OOD_index[ood_dataset]*args.APInum+node_id
            else:
                ood_dataset = OOD_list[args.fakefrom]
                model_id = OOD_index[ood_dataset] * args.APInum + node_id
            pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, args.pre_backbone, args.way_pretrain)
            select_dataset = ood_dataset
        elif temp <= args.oodP + args.fakeP:
            model_id=OOD_index['fake']*args.APInum+node_id
        else:
            pass
        tmp = torch.load(os.path.join(pretrained_path, 'model_specific_acc_{}.pth'.format(node_id)))
        if temp>args.oodP and temp<=args.oodP+args.fakeP:#fake attack
            pass
        else:
            teacher.load_state_dict(tmp['teacher'])
        specific = tmp['specific']
        specific = [i + bias[select_dataset] for i in specific]
        if temp <= args.oodP:
            specific = random.sample(range(dataset_classnum[args.dataset]),args.way_pretrain)
            specific = [i + bias[args.dataset] for i in specific]
        acc=tmp['acc']
        return teacher, specific, acc,select_dataset,model_id
    else:
        if args.dataset != 'mix':  # sh
            select_dataset=args.dataset
            random_pretrain = random.choice(teacher_list)
            teacher = get_model(args=args,type=random_pretrain,set_maml_value=False,arbitrary_input=False).cuda(args.device)
            node_id = random.randint(0, args.APInum - 1)
            pretrained_path = './pretrained/{}_{}/{}way/model'.format(args.dataset, random_pretrain, args.way_pretrain)
            temp = random.random()
            if temp <= args.oodP:
                if args.fakefrom == -1:
                    ood_dataset = random.choice(OOD_list)
                else:
                    ood_dataset = OOD_list[args.fakefrom]
                pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, random_pretrain,args.way_pretrain)
                select_dataset=ood_dataset
            elif temp <= args.oodP + args.fakeP:
                #pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(args.dataset, random_pretrain,args.way_pretrain)
                pass
            else:
                pass

            tmp = torch.load(os.path.join(pretrained_path, 'model_specific_acc_{}.pth'.format(node_id)))
            if temp > args.oodP and temp <= args.oodP + args.fakeP:
                pass
            else:
                teacher.load_state_dict(tmp['teacher'])
            specific = tmp['specific']
            specific = [i + bias[select_dataset] for i in specific]
            if temp <= args.oodP:
                specific = random.sample(range(dataset_classnum[args.dataset]), args.way_pretrain)
                specific = [i + bias[args.dataset] for i in specific]
            acc = tmp['acc']
            return teacher, specific, acc,select_dataset
        elif args.dataset == 'mix':  # mh
            select_dataset=None
            random_pretrain = random.choice(teacher_list)
            random_dataset = random.choice(dataset_list)
            select_dataset=random_dataset
            teacher = get_model(args=args,type=random_pretrain,set_maml_value=False,arbitrary_input=False).cuda(args.device)
            node_id = random.randint(0, args.APInum - 1)
            temp = random.random()
            if temp <= args.oodP:
                if args.fakefrom == -1:
                    ood_dataset = random.choice(OOD_list)
                else:
                    ood_dataset = OOD_list[args.fakefrom]
                pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, random_pretrain,args.way_pretrain)
                select_dataset=ood_dataset
            elif temp <= args.oodP + args.fakeP:
                #pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(random_dataset, random_pretrain,args.way_pretrain)
                pass
            else:
                pretrained_path = './pretrained/{}_{}/{}way/model'.format(random_dataset, random_pretrain,args.way_pretrain)
            tmp = torch.load(os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
            if temp > args.oodP and temp <= args.oodP + args.fakeP:
                pass
            else:
                teacher.load_state_dict(tmp['teacher'])
            specific = tmp['specific']
            specific = [i + bias[select_dataset] for i in specific]
            if temp <= args.oodP:
                specific = random.sample(range(dataset_classnum[args.dataset]), args.way_pretrain)
                specific = [i + bias[args.dataset] for i in specific]
            acc=tmp['acc']
            return teacher, specific, acc,select_dataset

def get_random_teacher_unique(args):
    args.APInum=math.ceil(dataset_classnum[args.dataset]/float(args.way_train))
    if args.pre_backbone != 'mix': #ss
        teacher = get_model(args=args,type=args.pre_backbone,set_maml_value=False,arbitrary_input=False).cuda(args.device)
        node_id = random.randint(0, args.APInum - 1)
        select_dataset=args.dataset
        pretrained_path = './pretrained/{}_{}/{}way/model_unique'.format(args.dataset, args.pre_backbone,args.way_pretrain)
        if temp <= args.oodP:
            if args.fakefrom == -1:
                ood_dataset = random.choice(OOD_list)
            else:
                ood_dataset = OOD_list[args.fakefrom]
            print('ood attack from', args.fakefrom)
            pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, args.pre_backbone, args.way_pretrain)
            select_dataset = ood_dataset
        elif temp <= args.oodP + args.fakeP:
            pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(args.dataset, args.pre_backbone,
                                                                           args.way_pretrain)
        else:
            pass
        tmp = torch.load(os.path.join(pretrained_path, 'model_specific_acc_{}.pth'.format(node_id)))
        teacher.load_state_dict(tmp['teacher'])
        specific = tmp['specific']
        specific = [i + bias[select_dataset] for i in specific]
        acc=tmp['acc']
        return teacher, specific, acc,select_dataset
    else:
        if args.dataset != 'mix':  # sh
            select_dataset=args.dataset
            random_pretrain = random.choice(teacher_list)
            teacher = get_model(args=args,type=random_pretrain,set_maml_value=False,arbitrary_input=False).cuda(args.device)
            node_id = random.randint(0, args.APInum - 1)
            pretrained_path = './pretrained/{}_{}/{}way/model'.format(args.dataset, random_pretrain, args.way_pretrain)
            temp = random.random()
            if temp <= args.oodP:
                if args.fakefrom == -1:
                    ood_dataset = random.choice(OOD_list)
                else:
                    ood_dataset = OOD_list[args.fakefrom]
                pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, random_pretrain,args.way_pretrain)
                select_dataset=ood_dataset
            elif temp <= args.oodP + args.fakeP:
                pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(args.dataset, random_pretrain,args.way_pretrain)
            else:
                pass
            tmp = torch.load(os.path.join(pretrained_path, 'model_specific_acc_{}.pth'.format(node_id)))
            teacher.load_state_dict(tmp['teacher'])
            specific = tmp['specific']
            specific = [i + bias[select_dataset] for i in specific]
            acc = tmp['acc']
            return teacher, specific, acc,select_dataset
        elif args.dataset == 'mix':  # mh
            select_dataset=None
            random_pretrain = random.choice(teacher_list)
            random_dataset = random.choice(dataset_list)
            select_dataset=random_dataset
            teacher = get_model(args=args,type=random_pretrain,set_maml_value=False,arbitrary_input=False).cuda(args.device)
            node_id = random.randint(0, args.APInum - 1)
            temp = random.random()
            if temp <= args.oodP:
                if args.fakefrom == -1:
                    ood_dataset = random.choice(OOD_list)
                else:
                    ood_dataset = OOD_list[args.fakefrom]
                pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, random_pretrain,args.way_pretrain)
                select_dataset=ood_dataset
            elif temp <= args.oodP + args.fakeP:
                pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(random_dataset, random_pretrain,args.way_pretrain)
            else:
                pretrained_path = './pretrained/{}_{}/{}way/model'.format(random_dataset, random_pretrain,args.way_pretrain)
            tmp = torch.load(os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
            teacher.load_state_dict(tmp['teacher'])
            specific = tmp['specific']
            specific = [i + bias[random_dataset] for i in specific]
            acc=tmp['acc']
            return teacher, specific, acc,select_dataset

def get_model_by_id(args,model_id):
    teacher = get_model(args=args, type=args.pre_backbone, set_maml_value=False, arbitrary_input=False).cuda(args.device)
    attribute=model_id//100
    node_id = model_id%100
    if attribute==0:
        pretrained_path = './pretrained/{}_{}/{}way/model'.format(args.dataset, args.pre_backbone, args.way_pretrain)
        select_dataset=args.dataset
    elif attribute==6:
        pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(args.dataset, args.pre_backbone, args.way_pretrain)
        select_dataset = args.dataset

    else:
        ood_dataset=OOD_list[attribute-1]
        pretrained_path = './pretrained/{}_{}/{}way/model'.format(ood_dataset, args.pre_backbone, args.way_pretrain)
        select_dataset = ood_dataset
    tmp = torch.load(os.path.join(pretrained_path, 'model_specific_acc_{}.pth'.format(node_id)))

    if attribute==6:  # fake attack
        pass
    else:
        teacher.load_state_dict(tmp['teacher'])
    specific = tmp['specific']
    specific = [i + bias[select_dataset] for i in specific]
    if attribute>=1 and attribute<=5:
        specific = random.sample(range(dataset_classnum[args.dataset]), args.way_pretrain)
        specific = [i + bias[args.dataset] for i in specific]
    acc = tmp['acc']
    return teacher, specific, acc, select_dataset, model_id



class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def data2supportquery(args,data):
    way = args.way_test
    num_sup = args.num_sup_test
    num_qur = args.num_qur_test
    label = torch.arange(way, dtype=torch.int16).repeat(num_qur+num_sup)
    label = label.type(torch.LongTensor)
    label = label.cuda()
    support=data[:way*num_sup]
    support_label=label[:way*num_sup]
    query=data[way*num_sup:]
    query_label=label[way*num_sup:]
    return support,support_label,query,query_label

NORMALIZE_DICT = {
    #'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'miniimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cub': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'eurosat':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'isic':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'chest':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cropdiseases':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'omniglot':dict(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426]),
    'mnist':dict(mean=[0.13066, 0.13066, 0.13066], std=[0.30131, 0.30131, 0.30131]),
    'tinyimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    'cub200': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_dogs': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_cars': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_64x64': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'svhn': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'tiny_imagenet': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'imagenet_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # for semantic segmentation
    'camvid': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'nyuv2': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor
class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)
def label_abs2relative(specific, label_abs):
    trans = dict()
    for relative, abs in enumerate(specific):
        trans[abs] = relative
    label_relative = []
    for abs in label_abs:
        label_relative.append(trans[abs.item()])
    return torch.LongTensor(label_relative)



def pretrain(args,specific,device):
    if args.dataset=='cifar100':
        train_dataset = Cifar100_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset)==args.way_pretrain*480, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        test_dataset = Cifar100_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == args.way_pretrain*120, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        channel=3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='miniimagenet':
        train_dataset = MiniImageNet_Specific(setname='meta_train', specific=specific, mode='train')
        assert len(train_dataset) == args.way_pretrain*480, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = MiniImageNet_Specific(setname='meta_train', specific=specific, mode='test')
        assert len(test_dataset) == args.way_pretrain*120, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='omniglot':
        train_dataset = Omniglot_Specific(setname='meta_train', specific=specific, mode='train')
        assert len(train_dataset) == 80, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=80, shuffle=True, num_workers=8,pin_memory=True)
        test_dataset = Omniglot_Specific(setname='meta_train', specific=specific, mode='test')
        assert len(test_dataset) == 20, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=20, shuffle=True, num_workers=8,pin_memory=True)
        channel = 3
        num_epoch = 30
        learning_rate = 0.01
    elif args.dataset=='mnist':
        train_dataset = Mnist_Specific(setname='meta_train',specific=specific)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        test_dataset = Mnist_Specific(setname='meta_test',specific=specific)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        channel = 3
        num_epoch = 30
        learning_rate = 0.01
    elif args.dataset=='cub':
        train_dataset = CUB_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset) == 2400, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        test_dataset = CUB_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == 600, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='flower':
        train_dataset = flower_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset) == 2400, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = flower_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == 600, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='cropdiseases':
        train_dataset = cropdiseases_Specific(setname='meta_train', specific=specific, mode='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = cropdiseases_Specific(setname='meta_train', specific=specific, mode='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 1
        learning_rate = 0.01
    elif args.dataset=='eurosat':
        train_dataset = eurosat_Specific(setname='meta_train', specific=specific, mode='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = eurosat_Specific(setname='meta_train', specific=specific, mode='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='isic':
        train_dataset = isic_Specific(setname='meta_train', specific=specific, mode='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = isic_Specific(setname='meta_train', specific=specific, mode='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='chest':
        train_dataset = chest_Specific(setname='meta_train', specific=specific, mode='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = chest_Specific(setname='meta_train', specific=specific, mode='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 1
        learning_rate = 0.01
    else:
        raise NotImplementedError
    set_maml(False)
    if args.pre_backbone=='conv4':
        teacher=Conv4(flatten=True, out_dim=args.way_pretrain, img_size=train_dataset.img_size,arbitrary_input=False,channel=channel).cuda(device)
        optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        #optimizer=torch.optim.SGD(params=teacher.parameters(),lr=learning_rate,momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[70], gamma=0.2)#70 default#[30, 50, 80]
    elif args.pre_backbone=='resnet18':
        teacher=ResNet18(flatten=True,out_dim=args.way_pretrain).cuda(device)
        #optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer=torch.optim.SGD(params=teacher.parameters(),lr=learning_rate,momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    elif args.pre_backbone=='resnet10':
        teacher = ResNet10(flatten=True, out_dim=args.way_pretrain).cuda(device)
        # optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    #train
    best_pre_model=None
    best_acc=None
    not_increase=0
    if args.fake_pretrain:
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        return teacher.state_dict(),test_acc
    for epoch in range(num_epoch):
        # train
        teacher.train()
        for batch_count, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            criteria = torch.nn.CrossEntropyLoss()
            loss = criteria(logits, relative_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 50)
            optimizer.step()
        lr_schedule.step()
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        if best_acc==None or best_acc<test_acc:
            best_acc=test_acc
            best_epoch=epoch
            best_pre_model=teacher.state_dict()
            not_increase=0
        else:
            not_increase=not_increase+1
            if not_increase==60:#7 for cifar and mini; 20 for omniglot
                print('early stop at:',best_epoch)
                break
        #print('epoch{}acc:'.format(epoch),test_acc,'best{}acc:'.format(best_epoch),best_acc)

    return best_pre_model,best_acc
def pretrains(args,num):
    if args.fake_pretrain == False:
        pretrained_path = './pretrained/{}_{}/{}way/model'.format(args.dataset, args.pre_backbone, args.way_pretrain)
    else:
        pretrained_path = './pretrained/{}_{}/{}way/model_fake'.format(args.dataset, args.pre_backbone,args.way_pretrain)
    logger = get_logger(pretrained_path, output=pretrained_path + '/' + 'log_pretrain.txt')
    timer=Timer()
    for i in range(num):
        specific=random.sample(range(dataset_classnum[args.dataset]),args.way_pretrain)
        teacher,acc=pretrain(args,specific,args.device)
        logger.info('id:{}, specific:{}, acc:{}'.format(i,specific,acc))
        torch.save({'teacher':teacher,'specific':specific,'acc':acc},os.path.join(pretrained_path,'model_specific_acc_{}.pth'.format(i)))
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((i+1) / (num)))
        )


def pretrains_unique(args):
    if args.fake_pretrain == False:
        pretrained_path = './pretrained/{}_{}/{}way/model_unique'.format(args.dataset, args.pre_backbone, args.way_pretrain)
    else:
        pretrained_path = './pretrained/{}_{}/{}way/model_fake_unique'.format(args.dataset, args.pre_backbone,args.way_pretrain)
    logger = get_logger(pretrained_path, output=pretrained_path + '/' + 'log_pretrain.txt')
    timer=Timer()
    num=math.ceil(dataset_classnum[args.dataset]/float(args.way_train))
    for i in range(num):
        if i!=12:
            continue
        specific=list(range(i*args.way_pretrain,min(dataset_classnum[args.dataset],i*args.way_pretrain+args.way_pretrain)))
        if len(specific)!=args.way_pretrain:
            need=args.way_pretrain-len(specific)
            for n in range(need):
                specific.append(n)
        print(i,':',specific)
        teacher,acc=pretrain(args,specific,args.device)
        logger.info('id:{}, specific:{}, acc:{}'.format(i,specific,acc))
        torch.save({'teacher':teacher,'specific':specific,'acc':acc},os.path.join(pretrained_path,'model_specific_acc_{}.pth'.format(i)))
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((i+1) / (num)))
        )

def get_transform(args,dataset=None):
    if dataset==None:
        dataset=args.dataset
    transform=None
    if dataset=='cifar100':
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    elif dataset=='miniimagenet':
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='omniglot':
        transform = transforms.Compose(
            [
                #transforms.Resize((28, 28)),
                lambda x: x.resize((28, 28)),
                lambda x: np.reshape(x, (28, 28, 1)),
                transforms.ToTensor(),
            ]
        )
    elif dataset=='cub':
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='flower':
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        raise NotImplementedError
    return transform




def get_transform_no_toTensor(args,dataset=None):
    if dataset==None:
        dataset=args.dataset
    transform = None
    if dataset=='cifar100':
        transform = transforms.Compose(
            [
                #transforms.Resize((32, 32)),
                transforms.RandomCrop(size=[32, 32], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    elif dataset=='miniimagenet':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=[32, 32], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='omniglot':
        transform = transforms.Compose(
            [
                #lambda x: x.resize((28, 28), padding=4),
                transforms.RandomCrop((28, 28), padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif dataset=='cub':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=[32, 32], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='flower':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=[32, 32], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return transform



def one_hot(label_list,class_num):
    temp_label=label_list.reshape(len(label_list),1)
    y_one_hot = torch.zeros(size=[len(label_list), class_num],device='cuda:0').scatter_(1, temp_label, 1)
    return y_one_hot

def shuffle_task(args,support,support_label,query,query_label):
    support_label_pair=list(zip(support,support_label))
    np.random.shuffle(support_label_pair)
    support,support_label=zip(*support_label_pair)
    support=torch.stack(list(support),dim=0).cuda(args.device)
    support_label=torch.tensor(list(support_label)).cuda(args.device)

    query_label_pair = list(zip(query, query_label))
    np.random.shuffle(query_label_pair)
    query, query_label = zip(*query_label_pair)
    query = torch.stack(list(query), dim=0).cuda(args.device)
    query_label = torch.tensor(list(query_label)).cuda(args.device)

    return support,support_label,query,query_label

def construct_model_pool(args,unique=True):
    model_path_list=[]
    model_attribute=[]
    #ID dataset
    if unique==True:
        normal_num=math.ceil(dataset_classnum[args.dataset]/float(args.way_pretrain))
        for i in range(normal_num):
            model_path_list.append('./pretrained/{}_{}/{}way/model_unique/model_specific_acc_{}.pth'.format(args.dataset, args.pre_backbone, args.way_pretrain,i))
            model_attribute.append(0)

        total_num = math.ceil(normal_num / (1.0 - args.oodP - args.fakeP))
        ood_fake_num = total_num - normal_num
        ood_num = math.ceil(total_num * args.oodP)
        fake_num = ood_fake_num - ood_num
        print('totoal:', total_num, 'normal:', normal_num, 'ood:', ood_num, 'fake:', fake_num)
        # OOD attack
        ood_ids = random.sample(list(range(args.APInum)), ood_num)
        for i,ood_id in enumerate(ood_ids):
            if args.fakefrom == -1:
                ood_dataset = OOD_list[i%len(OOD_list)]
            else:
                ood_dataset = OOD_list[args.fakefrom]
            print(ood_dataset)
            model_path_list.append(
                './pretrained/{}_{}/{}way/model/model_specific_acc_{}.pth'.format(ood_dataset, args.pre_backbone,
                                                                                  args.way_pretrain, ood_id))
            model_attribute.append(1)
    else:
        normal_num=args.APInum
        for i in range(normal_num):
            model_path_list.append('./pretrained/{}_{}/{}way/model/model_specific_acc_{}.pth'.format(args.dataset, args.pre_backbone, args.way_pretrain,i))
            model_attribute.append(0)

        total_num = math.ceil(normal_num / (1.0 - args.oodP - args.fakeP))
        ood_fake_num = total_num - normal_num
        ood_num = math.ceil(total_num * args.oodP)
        fake_num = ood_fake_num - ood_num
        print('totoal:', total_num, 'normal:', normal_num, 'ood:', ood_num, 'fake:', fake_num)
        # OOD attack
        if args.fakefrom==-1:
            ood_ids = random.choices(list(range(args.APInum)), k=ood_num)
        else:
            ood_ids = random.sample(list(range(args.APInum)), ood_num)
        for ood_id in ood_ids:
            if args.fakefrom == -1:
                ood_dataset = random.choice(OOD_list)
                # ood_dataset = OOD_list[i%len(OOD_list)]
            else:
                ood_dataset = OOD_list[args.fakefrom]
            print(ood_dataset)
            model_path_list.append(
                './pretrained/{}_{}/{}way/model/model_specific_acc_{}.pth'.format(ood_dataset, args.pre_backbone,
                                                                                  args.way_pretrain, ood_id))
            model_attribute.append(1)

    #fake attack
    fake_ids=random.sample(range(args.APInum),fake_num)
    for fake_id in fake_ids:
        model_path_list.append('./pretrained/{}_{}/{}way/model_fake/model_specific_acc_{}.pth'.format(args.dataset, args.pre_backbone, args.way_pretrain,fake_id))
        model_attribute.append(2)
    return model_path_list,model_attribute

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam.cpu())
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_data(xs, xq, lam):
    mixed_x = lam * xq + (1 - lam) * xs

    # mixed_x = xq.clone()
    # bbx1, bby1, bbx2, bby2 = rand_bbox(xq.size(), lam)
    #
    # mixed_x[:, :, bbx1:bbx2, bby1:bby2] = xs[:, :, bbx1:bbx2, bby1:bby2]
    #
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xq.size()[-1] * xq.size()[-2]))

    return mixed_x, lam