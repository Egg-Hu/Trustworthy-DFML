import argparse
import math
import os
from datetime import datetime
import random
import shutil
import numpy as np
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
from torch import nn

from method.load import LOAD
from method.average import Average
from method.reject import Reject
from method.weight import Weight
from method.online import Online
from method.preinversiondfml import PRE
from method.purer6 import PURER
from method.taco6 import TACO
from method.tmeta import TMETA
from method.gene import GENE
from method.random import RANDOM
from method.bidfmkd import BiDFMKD
from method.maml import MAML
from method.maml_S import MAMLS
from synthesis.deepinversion import DeepInvSyntheiszer
from synthesis._utils import save_image_batch
from logger import get_logger
from generator import Generator
from black_box_tool import get_model, Normalizer, NORMALIZE_DICT, get_transform, \
    get_transform_no_toTensor, \
    label_abs2relative, get_dataloader, data2supportquery, Timer, setup_seed, compute_confidence_interval, \
    pretrains, set_maml, pretrain, pretrains_unique
from maml import MamlKD,Maml
from synthesis.contrastive import Synthesizer

parser = argparse.ArgumentParser(description='maincmi')
#basic
parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar100/miniimagenet/omniglot')
parser.add_argument('--testdataset', type=str, default='cifar100', help='cifar100/miniimagenet/omniglot')
parser.add_argument('--val_interval',type=int, default=2000)
parser.add_argument('--save_interval',type=int, default=2000)
parser.add_argument('--episode_batch',type=int, default=4)
#maml
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4', help='conv4/resnet34/resnet18')
parser.add_argument('--episode_train', type=int, default=240000)
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
#taco
parser.add_argument('--candidate_size', type=int, default=6,help='num of candidate tasks, only for taco')
parser.add_argument('--scheduleScheduler', action='store_true',default=False)
parser.add_argument('--scheduleWeight', action='store_true',default=False)
parser.add_argument('--rewardType', type=int, default=0)
parser.add_argument('--earlyStop', type=int, default=0)
parser.add_argument('--nomodelpool', action='store_true',default=False)
parser.add_argument('--notmeta', action='store_true',default=False)
parser.add_argument('--warmup', type=int, default=3000)
parser.add_argument('--noteacher', action='store_true',default=False)
#method
parser.add_argument('--method', type=str, default='bidfmkd', help='bidfmkd/maml/protonet')
parser.add_argument('--teacherMethod', type=str, default='maml', help='maml/protonet')
#bidfmkd
parser.add_argument('--num_sup_kd', type=int, default=30)
parser.add_argument('--num_qur_kd', type=int, default=30)
parser.add_argument('--inner_update_num_kd', type=int, default=10)
#dfmeta
parser.add_argument('--inversionMethod', type=str, default='deepinv')
parser.add_argument('--way_pretrain', type=int, default=5, help='way')
parser.add_argument('--APInum', type=int, default=100)
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet34/resnet18')
parser.add_argument('--noBnLoss', action='store_true',default=False)
parser.add_argument('--generate_interval', type=int, default=200)
parser.add_argument('--generate_iterations', type=int, default=200)
parser.add_argument('--Glr', type=float, default=0.001)
parser.add_argument('--Slr', type=float, default=0.1)
parser.add_argument('--dynamicSlr', action='store_true',default=False)
parser.add_argument('--oodP', type=float, default=0.0)
parser.add_argument('--fakeP', type=float, default=0.0)
parser.add_argument('--fakefrom', type=int, default=-1)
parser.add_argument('--preGenerate', action='store_true',default=False)
parser.add_argument('--nomemory', action='store_true',default=False)
parser.add_argument('--time', action='store_true',default=False)
parser.add_argument('--defense', action='store_true',default=False)
parser.add_argument('--defensew', action='store_true',default=False)
parser.add_argument('--defenses', action='store_true',default=False)
parser.add_argument('--dynamicSpecific', action='store_true',default=False)
parser.add_argument('--unique', action='store_true',default=False)
parser.add_argument('--maxbatch', type=int, default=20)
#zero-order optimization
parser.add_argument('--ZO', action='store_true',default=False)
parser.add_argument('--mu', type=float, default=0.005)
parser.add_argument('--q', type=int, default=100)
parser.add_argument('--numsplit', type=int, default=5)
#else
parser.add_argument('--extra', type=str, default='')
#pretrain
parser.add_argument('--pretrain', action='store_true',default=False)
parser.add_argument('--fake_pretrain', action='store_true',default=False)
parser.add_argument('--evaluate', action='store_true',default=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
setup_seed(2022)
args.device=torch.device('cuda:{}'.format(args.gpu))
########
if args.pretrain or args.fake_pretrain:
    pretrains(args,args.APInum)
    #pretrains_unique(args)
elif args.evaluate:
    pass
else:
    method_dict = dict(
                bidfmkd = BiDFMKD,
                random=RANDOM,
                gene=GENE,
                maml=MAML,
                mamlS=MAMLS,
                tmeta=TMETA,
                taco=TACO,
                purer=PURER,
                pre=PRE,
                online=Online,
                weight=Weight,
                reject=Reject,
                average=Average,
                load=LOAD,
    )
    method=method_dict[args.method](args)
    method.train_loop()
