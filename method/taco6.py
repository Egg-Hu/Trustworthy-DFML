
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.distributions import Beta

import random
import shutil
import sys
from collections import OrderedDict
from copy import deepcopy
from os import listdir


sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch.nn.functional as F

import torch
import torch.nn as nn
from scheduler import Scheduler
from dataset.cifar100 import Cifar100
from black_box_tool import get_transform_no_toTensor, get_transform, Normalizer, NORMALIZE_DICT, \
    label_abs2relative, get_model_by_id, mixup_data
from generator import Generator,GeneratorCon
from synthesis import Synthesizer, DeepInvSyntheiszer
from synthesis.concontrastive import ConSynthesizer
from tensorboardX import SummaryWriter
from black_box_tool import set_maml, get_model, get_random_teacher, kldiv, data2supportquery, \
    compute_confidence_interval, get_dataloader, Timer,shuffle_task, bias , construct_model_pool,dataset_classnum
from logger import get_logger
from synthesis._utils import save_image_batch
from torch.distributions.categorical import Categorical



class TACO(nn.Module):
    def __init__(self,args):
        super(TACO, self).__init__()
        self.args=args
        # file
        feature1 = '{}_{}_{}'.format(args.method, args.teacherMethod, args.inversionMethod)
        if self.args.dataset==self.args.testdataset:
            feature2_1 = '{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset,args.pre_backbone,args.backbone, args.APInum, args.oodP,args.fakeP, args.fakefrom)
        else:
            feature2_1 = '{}/{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset, args.testdataset,args.pre_backbone,args.backbone, args.APInum, args.oodP,args.fakeP, args.fakefrom)
        if args.teacherMethod == 'maml':
            feature2_2 = '{}wPre_{}S_{}Q_{}kds_{}kdq_{}stepkd_{}steptrain_{}steptest{}innerlr_{}outerlr_{}batch_{}Ginterval_{}Git_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.num_sup_kd, args.num_qur_kd,
                args.inner_update_num_kd, args.inner_update_num,
                args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,
                args.generate_interval, args.generate_iterations, args.Glr)
        elif args.teacherMethod == 'protonet':
            feature2_2 = '{}wPre_{}S_{}Q_{}lr_{}batch_{}InvM_{}Ginterval_{}Git_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.outer_lr, args.episode_batch,
                args.inversionMethod,  args.generate_interval, args.generate_iterations, args.Glr)
        elif args.teacherMethod == 'anil':
            feature2_2 = '{}wPre_{}S_{}Q_{}kds_{}kdq_{}stepkd_{}steptrain_{}steptest{}innerlr_{}outerlr_{}batch_{}Ginterval_{}Git_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.num_sup_kd, args.num_qur_kd,
                args.inner_update_num_kd, args.inner_update_num,
                args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,
                args.generate_interval, args.generate_iterations, args.Glr)
        feature2 = feature2_1 + '_' + feature2_2
        if self.args.preGenerate==True and self.args.scheduleScheduler:
            feature2 = feature2 + '_PreGen'
        if self.args.earlyStop==1:
            feature2 = feature2 + '_EarlyStopByTrainLoss'
        elif self.args.earlyStop==2:
            feature2 = feature2 + '_EarlyStopByValAcc'
        elif self.args.earlyStop==0:
            feature2 = feature2 + '_EarlyStopByFix'
        if self.args.scheduleScheduler:
            feature2=feature2+'_Scheduler'
        elif self.args.scheduleWeight:
            feature2=feature2+'_Weight'
        if (self.args.scheduleScheduler or self.args.scheduleWeight):
            if self.args.rewardType==0:
                feature2 = feature2 + '_Reward0'
            elif self.args.rewardType==1:
                feature2 = feature2 + '_Reward1'
            elif self.args.rewardType==2:
                feature2 = feature2 + '_Reward2'
            feature2 = feature2 + '_CandidateSize{}'.format(self.args.candidate_size)
            feature2 = feature2 + '_Slr{}'.format(self.args.Slr)
        feature2 = feature2 + '_maxBatch{}'.format(self.args.maxbatch)
        if self.args.noteacher:
            feature2 = feature2 + '_NoTeacher'
        feature2 = feature2 + '_{}'.format(args.extra)
        if args.approx:
            feature2 = feature2 + '_1Order'
        if (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==False and self.args.scheduleWeight==False):
            self.checkpoints_path = './checkpoints_taco6/'+'ATTACK/'+ feature1 + '/' + feature2
        elif (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==True or self.args.scheduleWeight==True):
            self.checkpoints_path = './checkpoints_taco6/' + 'DEFENSE/' + feature1 + '/' + feature2
        elif self.args.oodP==0 or self.args.fakeP==0:
            self.checkpoints_path = './checkpoints_taco6/' + 'SAFE/' + feature1 + '/' + feature2
        self.writer_path = os.path.join(self.checkpoints_path, 'writer')
        if os.path.exists(self.writer_path):
            shutil.rmtree(self.writer_path)
        os.makedirs(self.writer_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.logger = get_logger(feature1 + '/' + feature2, output=self.checkpoints_path + '/' + 'log.txt')
        _, self.val_loader, self.test_loader = get_dataloader(self.args)
        # meta model
        if self.args.teacherMethod == 'maml':
            set_maml(True)
            self.model = get_model(args=args, type=args.backbone, set_maml_value=True, arbitrary_input=False).cuda(self.args.device)
            set_maml(False)
            self.model.trunk[-1].bias.data.fill_(0)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.outer_lr)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.args.teacherMethod == 'protonet':
            set_maml(True)
            self.model = get_model(args=args, type=args.backbone, set_maml_value=True, arbitrary_input=False).cuda(self.args.device)
            set_maml(False)
            self.model.trunk[-1].bias.data.fill_(0)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.outer_lr)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.args.teacherMethod=='anil':
            set_maml(True)
            self.model = get_model(args=args, type=args.backbone, set_maml_value=True, arbitrary_input=False).cuda(
                self.args.device)
            set_maml(False)
            self.model.trunk[-1].bias.data.fill_(0)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.outer_lr)
            self.loss_fn = nn.CrossEntropyLoss()

        #synthesizer
        nz = 256
        if args.inversionMethod=='concmi':
            generator = GeneratorCon(nz=nz, ngf=64, img_size=args.img_size, nc=args.channel).cuda(args.device)
        else:
            generator = Generator(nz=nz, ngf=64, img_size=args.img_size, nc=args.channel).cuda(args.device)
        self.transform_no_toTensor = get_transform_no_toTensor(args)
        self.transform = get_transform(args)
        if (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==False and self.args.scheduleWeight==False):
            datapool_path = self.datapool_path = './datapool_taco6/' + 'ATTACK/' + os.path.join(feature1, feature2)
        elif (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==True or self.args.scheduleWeight==True):
            datapool_path = self.datapool_path = './datapool_taco6/' + 'DEFENSE/' + os.path.join(feature1, feature2)
        elif self.args.oodP==0 or self.args.fakeP==0:
            datapool_path = self.datapool_path = './datapool_taco6/' + 'SAFE/' + os.path.join(feature1, feature2)

        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print('remove')
        if os.path.exists('./tsne'):
            shutil.rmtree('./tsne')
            os.makedirs('./tsne')
        #os.makedirs('./datapool/' + os.path.join(feature1, feature2), exist_ok=True)

        if self.args.preGenerate:
            max_batch_per_class = self.args.maxbatch
            self.preGenerate_path = './preGenerate/' + os.path.join(feature1, feature2)
        else:
            max_batch_per_class = self.args.maxbatch
        if args.inversionMethod == 'cmi':
            self.synthesizer = Synthesizer(args, None, None, generator, nz=nz, num_classes=500,
                                      img_size=(args.channel, args.img_size, args.img_size),
                                      iterations=args.generate_iterations, lr_g=args.Glr,
                                      synthesis_batch_size=30,
                                      bn=1.0, oh=1.0, adv=0.0,
                                      save_dir=datapool_path, transform=self.transform,
                                      transform_no_toTensor=self.transform_no_toTensor,
                                      device=args.gpu, c_abs_list=None, max_batch_per_class=max_batch_per_class)
        elif args.inversionMethod == 'deepinv':
            self.synthesizer = DeepInvSyntheiszer(args, None, None, img_size=(args.channel, args.img_size, args.img_size),
                                             iterations=args.generate_iterations, lr_g=args.Glr,
                                             synthesis_batch_size=30,
                                             adv=0.0, bn=0.01, oh=1, tv=1e-4, l2=0.0,
                                             save_dir=datapool_path, transform=self.transform,
                                             normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]), device=args.gpu,
                                             num_classes=500, c_abs_list=None,
                                             max_batch_per_class=max_batch_per_class)
        elif args.inversionMethod=='concmi':
            self.synthesizer = ConSynthesizer(args, None, None, generator, nz=nz, num_classes=500,
                                           img_size=(args.channel, args.img_size, args.img_size),
                                           iterations=args.generate_iterations, lr_g=args.Glr,
                                           synthesis_batch_size=30,
                                           bn=1.0, oh=1.0, adv=0.0,
                                           save_dir=datapool_path, transform=self.transform,
                                           transform_no_toTensor=self.transform_no_toTensor,
                                           device=args.gpu, c_abs_list=None, max_batch_per_class=max_batch_per_class)
        names_weights_copy, indexes=self.get_differential_parameter_dict(params=self.model.named_parameters())

        self.model_pool_path, self.model_pool_attribute = construct_model_pool(args=self.args, unique=False)

        if self.args.scheduleScheduler:
            self.scheduler=Scheduler(args=self.args,N=len(names_weights_copy),grad_indexes=indexes,use_deepsets=True).cuda(self.args.device)
            self.scheduler_optimizer = torch.optim.Adam(self.scheduler.parameters(), lr=args.Slr)
        elif self.args.scheduleWeight:
            self.model_pool_weight=torch.nn.Parameter(torch.ones(len(self.model_pool_path)).cuda(self.args.device)).requires_grad_()
            self.scheduler_optimizer = torch.optim.Adam([self.model_pool_weight], lr=args.Slr)

        self.model_pool_specific = []
        for i, path in enumerate(self.model_pool_path):
            tmp = torch.load(path)
            specific = tmp['specific']
            specific = [i + bias[self.args.dataset] for i in specific]
            if self.model_pool_attribute[i] == 1:
                specific = random.sample(range(dataset_classnum[self.args.dataset]), self.args.way_pretrain)
                specific = [s + bias[self.args.dataset] for s in specific]
            self.model_pool_specific.append(specific)

        if self.args.scheduleScheduler:
            print('model pool data start')
            if os.path.exists('./model_pool_data_taco6_mini/' + os.path.join(feature1, feature2)):
                self.model_pool_data = torch.load('./model_pool_data_taco6_mini/' + os.path.join(feature1, feature2) + '/model_pool_data.pth')
            else:
                self.model_pool_data=[]
                for i, path in enumerate(self.model_pool_path):
                    tmp = torch.load(path)
                    teacher = get_model(args=self.args, type=self.args.pre_backbone, set_maml_value=False,
                                        arbitrary_input=False).cuda(self.args.device)
                    teacher.load_state_dict(tmp['teacher'])
                    teacher.eval()
                    #specific=self.get_specific(i)
                    specific=self.model_pool_specific[i]
                    self.synthesizer.teacher = teacher
                    self.synthesizer.c_abs_list = specific

                    support_query_list=[]
                    support_query_label_relative_list=[]
                    for g_it in range(1):
                        support_query_tensor = self.synthesizer.synthesize(
                            targets=torch.LongTensor((list(range(len(specific)))) * (30)), student=None,
                            mode='all', c_num=len(specific),add=False)
                        support_query = self.transform_no_toTensor(support_query_tensor)
                        support_query_label_relative = torch.LongTensor((list(range(len(specific)))) * (30)).cuda(self.args.device)
                        support_query_list.append(support_query)
                        support_query_label_relative_list.append(support_query_label_relative)


                    support_query=torch.cat(support_query_list,dim=0)
                    support_query_label_relative=torch.cat(support_query_label_relative_list,dim=0)
                    self.model_pool_data.append((support_query,support_query_label_relative))
                os.makedirs('./model_pool_data_taco6_mini/' + os.path.join(feature1, feature2), exist_ok=True)
                torch.save(self.model_pool_data, './model_pool_data_taco6_mini/' + os.path.join(feature1, feature2) + '/model_pool_data.pth')
            print('model pool data end')

    def get_teacher(self,id):
        tmp = torch.load(self.model_pool_path[id])
        teacher = get_model(args=self.args, type=self.args.pre_backbone, set_maml_value=False,
                            arbitrary_input=False).cuda(self.args.device)
        teacher.load_state_dict(tmp['teacher'])
        teacher.eval()
        return teacher
    def forward(self,x):
        scores  = self.model(x)
        return scores
    def tmeta_loss(self,teacher,support,support_label,query,query_label,way=None):
        #teacher_meta_learning
        if self.args.teacherMethod == 'maml':
            # inner
            self.model.zero_grad()
            fast_parameters = list(self.model.parameters())
            for weight in self.model.parameters():
                weight.fast = None
            for task_step in range(self.args.inner_update_num_kd):
                s_logits = self.forward(support)
                with torch.no_grad():
                    t_logits = teacher(support)
                loss_inner = kldiv(s_logits, t_logits.detach())
                #loss_inner = F.kl_div(F.softmax(s_logits,dim=-1), F.softmax(t_logits.detach(),dim=-1))
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
                if self.args.approx == True:
                    grad = [g.detach() for g in grad]
                fast_parameters = []
                for k, weight in enumerate(self.model.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[k]
                    fast_parameters.append(weight.fast)
            # outer
            s_logits = self.model(query)
            with torch.no_grad():
                t_logits = teacher(query)
            loss_outer = kldiv(s_logits, t_logits.detach())
            for weight in self.model.parameters():
                weight.fast = None
        elif self.args.teacherMethod=='protonet':
            if way==None:
                way=self.args.way_train
            z_support = self.model.getFeature(support)
            z_query = self.model.getFeature(query)
            z_support = z_support.contiguous().view(way* self.args.num_sup_kd, -1)
            z_query = z_query.contiguous().view(way * self.args.num_qur_kd, -1)
            z_support = z_support.contiguous()
            protos = []
            for c in range(way):
                protos.append(z_support[support_label == c].mean(0))
            z_proto = torch.stack(protos, dim=0)
            z_query = z_query.contiguous().view(way * self.args.num_qur_kd, -1)
            dists = euclidean_dist(z_query, z_proto)
            s_logits = -dists
            with torch.no_grad():
                t_logits = teacher(query)
            if way!=self.args.way_pretrain:
                t_logits=t_logits[:,:way]
                print(t_logits.shape)
            loss_outer = kldiv(s_logits, t_logits.detach())
        elif self.args.teacherMethod=='anil':
            # inner
            self.model.zero_grad()
            #print(OrderedDict(self.model.named_parameters()).keys())
            fast_parameters = list(self.model.trunk[-1].parameters())
            for weight in self.model.parameters():
                weight.fast = None
            for task_step in range(self.args.inner_update_num_kd):
                s_logits = self.forward(support)
                with torch.no_grad():
                    t_logits = teacher(support)
                loss_inner = kldiv(s_logits, t_logits.detach())
                # loss_inner = F.kl_div(F.softmax(s_logits,dim=-1), F.softmax(t_logits.detach(),dim=-1))
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
                if self.args.approx == True:
                    grad = [g.detach() for g in grad]
                fast_parameters = []
                for k, weight in enumerate(self.model.trunk[-1].parameters()):
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[k]
                    fast_parameters.append(weight.fast)
            # outer
            s_logits = self.model(query)
            with torch.no_grad():
                t_logits = teacher(query)
            loss_outer = kldiv(s_logits, t_logits.detach())
            for weight in self.model.parameters():
                weight.fast = None
        return loss_outer

    def meta_loss(self, support, support_label,query,query_label,way=None):
        if self.args.teacherMethod == 'maml':
            #inner
            self.model.zero_grad()
            fast_parameters = list(self.model.parameters()) # the first gradient calcuated in line 45 is based on original weight
            for weight in self.model.parameters():
                weight.fast = None
            for task_step in range(self.args.inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label)
                grad = torch.autograd.grad(loss_inner, fast_parameters,create_graph=True)  # build full graph support gradient of gradient
                if self.args.approx:
                    grad = [g.detach() for g in grad]  # do not calculate gradient of gradient if using first order approximation
                fast_parameters = []
                for k, weight in enumerate(self.model.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append(weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
            #outer
            scores = self.forward(query)
            loss_outer = self.loss_fn(scores, query_label)
            for weight in self.model.parameters():
                weight.fast = None
        elif self.args.teacherMethod=='protonet':
            if way==None:
                way=self.args.way_train
            z_support = self.model.getFeature(support)
            z_query = self.model.getFeature(query)
            z_support = z_support.contiguous().view(way * self.args.num_sup_train, -1)
            z_query = z_query.contiguous().view(way * self.args.num_qur_train, -1)
            z_support = z_support.contiguous()
            protos = []
            for c in range(way):
                protos.append(z_support[support_label == c].mean(0))
            z_proto = torch.stack(protos, dim=0)
            z_query = z_query.contiguous().view(way * self.args.num_qur_train, -1)
            dists = euclidean_dist(z_query, z_proto)
            score = -dists
            loss_outer = self.loss_fn(score, query_label)
        elif self.args.teacherMethod=='anil':
            # inner
            self.model.zero_grad()
            fast_parameters = list(self.model.trunk[-1].parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.model.parameters():
                weight.fast = None
            for task_step in range(self.args.inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label)
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)  # build full graph support gradient of gradient
                if self.args.approx:
                    grad = [g.detach() for g in grad]  # do not calculate gradient of gradient if using first order approximation
                fast_parameters = []
                for k, weight in enumerate(self.model.trunk[-1].parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append(weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
            # outer
            scores = self.forward(query)
            loss_outer = self.loss_fn(scores, query_label)
            for weight in self.model.parameters():
                weight.fast = None
        return loss_outer

    def train_loop(self):
        # # prepare val
        # print('prepare val data start!')
        # valset = Cifar100(setname='meta_val', augment=False)
        # val_data = []
        # val_label_dataset = []
        # for i in range(len(valset)):
        #     data_i, label_i = valset.__getitem__(i)
        #     val_data.append(data_i)
        #     val_label_dataset.append(label_i)
        # val_data = torch.stack(val_data, dim=0)
        # val_label_dataset = torch.LongTensor(val_label_dataset)
        # print('prepare val data end!')



        timer = Timer()
        with SummaryWriter(self.writer_path) as writer:
            test_acc_max = 0
            max_it = 0
            max_pm = 0
            moving_avg_reward = 40
            schedule_id=0
            e_minLoss=None
            e_maxValAcc=None
            tmeta_idicator=True
            self.long_reward=[]
            self.sub_long_reward = []
            for it_id in range(1, self.args.episode_train//self.args.episode_batch + 1):
                # if self.args.earlyStop == 0:
                #     if it_id==1 or (it_id%self.args.generate_interval)==0:
                #         tmeta_idicator=True
                # if self.args.earlyStop==1:
                #     raise NotImplementedError
                # if self.args.earlyStop == 2:
                #     raise NotImplementedError

                # if it_id==1:
                #     add_memory=True
                # elif (it_id<=50) and it_id%1==0:
                #     add_memory=True
                # elif (it_id>50 and it_id<=2000) and it_id%10==0:
                #     add_memory=True
                # elif (it_id>2000 and it_id<=4000) and it_id%50==0:
                #     add_memory=True
                # elif (it_id>4000 and it_id<=10000) and it_id%100==0:
                #     add_memory=True
                # elif (it_id>20000) and it_id%200==0:
                #     add_memory=True
                # else:
                #     add_memory=False
                if it_id == 1:
                    tmeta_idicator = True
                    add_memory = True
                if add_memory==True:
                    if self.args.scheduleScheduler:
                        candidate_ids = random.sample(range(len(self.model_pool_path)),self.args.candidate_size)
                    elif self.args.scheduleWeight:
                        candidate_ids = random.sample(range(len(self.model_pool_path)),self.args.candidate_size)
                    if self.args.scheduleScheduler:
                        #scheduler1 scheduler
                        # candidate
                        candidate_specific=[]
                        task1_task2_losses = []
                        train_val_losses = []
                        candidate_losses=[]

                        for candidate_id in candidate_ids:
                            # generate support
                            #teacher=self.model_pool_teacher[candidate_id]
                            teacher=self.get_teacher(candidate_id)
                            #candidate_specific.append(self.get_specific(candidate_id))
                            candidate_specific.append(self.model_pool_specific[candidate_id])
                            teacher.eval()
                            data,label_relative=self.model_pool_data[candidate_id][0],self.model_pool_data[candidate_id][1]
                            support,support_label_relative,query,query_label_relative=get_random_task_from_data2(args=self.args,data=data,label_relative=label_relative)

                            candidate_losses.append(self.meta_loss(support=support, support_label=support_label_relative, query=query,query_label=query_label_relative).detach())


                            support1, support_label_relative1, query1, query_label_relative1 = get_random_task_from_data2(args=self.args, data=data,label_relative=label_relative)
                            support2, support_label_relative2, query2, query_label_relative2 = get_random_task_from_data2(args=self.args, data=data,label_relative=label_relative)
                            support1, support_label_relative1, query1, query_label_relative1 = shuffle_task(args=self.args, support=support1, support_label=support_label_relative1,query=query1, query_label=query_label_relative1)
                            support2, support_label_relative2, query2, query_label_relative2 = shuffle_task(args=self.args, support=support2, support_label=support_label_relative2,query=query2, query_label=query_label_relative2)
                            task1_task2_losses.append((self.meta_loss(support=support1,
                                                                      support_label=support_label_relative1,
                                                                      query=query1,
                                                                      query_label=query_label_relative1),
                                                       self.meta_loss(support=support2,
                                                                      support_label=support_label_relative2,
                                                                      query=query2,
                                                                      query_label=query_label_relative2)))
                            for val_batch in self.val_loader:
                                data_val, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(
                                    self.args.device)
                                support_val, support_label_relative_val, query_val, query_label_relative_val = data2supportquery(
                                    self.args, data_val)
                                break
                            support_train, support_label_relative_train, query_train, query_label_relative_train = get_random_task_from_data2(args=self.args, data=data, label_relative=label_relative)
                            support_val, support_label_relative_val, query_val, query_label_relative_val = shuffle_task(
                                args=self.args, support=support_val, support_label=support_label_relative_val,
                                query=query_val,
                                query_label=query_label_relative_val)
                            support_train, support_label_relative_train, query_train, query_label_relative_train = shuffle_task(
                                args=self.args, support=support_train, support_label=support_label_relative_train,
                                query=query_train,
                                query_label=query_label_relative_train)
                            train_val_losses.append((self.meta_loss(support=support_train,
                                                                    support_label=support_label_relative_train,
                                                                    query=query_train,
                                                                    query_label=query_label_relative_train),
                                                     self.meta_loss(support=support_val,
                                                                    support_label=support_label_relative_val,
                                                                    query=query_val,
                                                                    query_label=query_label_relative_val)))

                        # schedule_model
                        _, _, all_task_weight = self.scheduler.get_weight(task_losses=candidate_losses,
                                                                          task1_task2_losses=task1_task2_losses,
                                                                          train_val_losses=train_val_losses,
                                                                          model=self.model, pt=int(it_id / (self.args.episode_train // self.args.episode_batch) * 100),
                                                                          model_ids=candidate_ids, detach=False)

                        torch.cuda.empty_cache()
                        all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)
                        selected_tasks_idx = self.scheduler.sample_task(all_task_prob, self.args.episode_batch,replace=0)
                        task_data = []
                        for task_idx in selected_tasks_idx:
                            teacher=self.get_teacher(candidate_ids[task_idx])
                            specific=candidate_specific[task_idx]
                            teacher.eval()
                            self.synthesizer.teacher = teacher
                            self.synthesizer.c_abs_list = specific

                            support_query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                                (list(range(len(specific)))) * (self.args.num_sup_kd + self.args.num_qur_kd)),student=None, mode='all',c_num=len(specific))
                            support_query = self.transform_no_toTensor(support_query_tensor)
                            support_label_relative = torch.LongTensor(
                                (list(range(len(specific)))) * self.args.num_sup_kd).cuda(self.args.device)
                            query_label_relative = torch.LongTensor(
                                (list(range(len(specific)))) * self.args.num_qur_kd).cuda(self.args.device)
                            support = support_query[:len(specific) * self.args.num_sup_kd]
                            query = support_query[len(specific) * self.args.num_sup_kd:]

                            # support_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                            #     (list(range(len(specific)))) * (self.args.num_sup_kd)),
                            #     student=None, mode='support', c_num=len(specific))
                            # support = self.transform_no_toTensor(support_tensor)
                            # support_label_relative = torch.LongTensor(
                            #     (list(range(len(specific)))) * self.args.num_sup_kd).cuda(self.args.device)
                            # query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                            #     (list(range(len(specific)))) * (self.args.num_qur_kd)),
                            #     student=None, mode='query', c_num=len(specific))
                            # query = self.transform_no_toTensor(query_tensor)
                            # query_label_relative = torch.LongTensor(
                            #     (list(range(len(specific)))) * self.args.num_qur_kd).cuda(self.args.device)

                            task_data.append((support,support_label_relative,query,query_label_relative))

                            # loss_outer = self.tmeta_loss(teacher=teacher, support=support, support_label=support_label_relative, query=query,query_label=query_label_relative)
                            # task_losses.append(loss_outer)
                        # meta_batch_loss = torch.mean(torch.stack(task_losses))
                    elif self.args.scheduleWeight:
                        candidate_weight = self.model_pool_weight[candidate_ids]
                        candidate_prob = torch.softmax(candidate_weight.reshape(-1), dim=-1)
                        candidate_prob = candidate_prob.detach().cpu().numpy()
                        selected_tasks_idx = np.random.choice(range(len(candidate_ids)),
                                                              p=candidate_prob / np.sum(candidate_prob),
                                                              size=self.args.episode_batch, replace=False)
                        task_data = []
                        for task_idx in selected_tasks_idx:
                            #teacher=self.model_pool_teacher[candidate_ids[task_idx]]
                            teacher = self.get_teacher(candidate_ids[task_idx])
                            #specific=self.get_specific(candidate_ids[task_idx])
                            specific=self.model_pool_specific[candidate_ids[task_idx]]
                            teacher.eval()

                            self.synthesizer.teacher = teacher
                            self.synthesizer.c_abs_list = specific

                            # support_query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * (self.args.num_sup_kd + self.args.num_qur_kd)),student=None, mode='all',c_num=len(specific))
                            # support_query = self.transform_no_toTensor(support_query_tensor)
                            # support_label_relative = torch.LongTensor((list(range(len(specific)))) * self.args.num_sup_kd).cuda(self.args.device)
                            # query_label_relative = torch.LongTensor((list(range(len(specific)))) * self.args.num_qur_kd).cuda(self.args.device)
                            # support = support_query[:len(specific) * self.args.num_sup_kd]
                            # query = support_query[len(specific) * self.args.num_sup_kd:]

                            support_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                                (list(range(len(specific)))) * (self.args.num_sup_kd)),
                                student=None, mode='support', c_num=len(specific))
                            support = self.transform_no_toTensor(support_tensor)
                            support_label_relative = torch.LongTensor(
                                (list(range(len(specific)))) * self.args.num_sup_kd).cuda(self.args.device)
                            query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                                (list(range(len(specific)))) * (self.args.num_qur_kd)),
                                student=None, mode='query', c_num=len(specific))
                            query = self.transform_no_toTensor(query_tensor)
                            query_label_relative = torch.LongTensor(
                                (list(range(len(specific)))) * self.args.num_qur_kd).cuda(self.args.device)

                            task_data.append((support,support_label_relative,query,query_label_relative))

                    else:#no schedule
                        task_data = []
                        for _ in range(self.args.episode_batch):
                            # generate support
                            t_id=random.randint(0, len(self.model_pool_path) - 1)
                            #teacher=self.model_pool_teacher[t_id]
                            teacher = self.get_teacher(t_id)
                            #specific=self.get_specific(t_id)
                            specific=self.model_pool_specific[t_id]
                            teacher.eval()
                            self.synthesizer.teacher = teacher
                            self.synthesizer.c_abs_list = specific
                            # support_query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * (self.args.num_sup_kd + self.args.num_qur_kd)),student=None, mode='all',c_num=len(specific))
                            # support_query = self.transform_no_toTensor(support_query_tensor)
                            # support_label_relative = torch.LongTensor((list(range(len(specific)))) * self.args.num_sup_kd).cuda(self.args.device)
                            # query_label_relative = torch.LongTensor((list(range(len(specific)))) * self.args.num_qur_kd).cuda(self.args.device)
                            # support = support_query[:len(specific) * self.args.num_sup_kd]
                            # query = support_query[len(specific) * self.args.num_sup_kd:]

                            support_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                                (list(range(len(specific)))) * (self.args.num_sup_kd)),
                                student=None, mode='support', c_num=len(specific))
                            support = self.transform_no_toTensor(support_tensor)
                            support_label_relative = torch.LongTensor(
                                (list(range(len(specific)))) * self.args.num_sup_kd).cuda(self.args.device)
                            query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor(
                                (list(range(len(specific)))) * (self.args.num_qur_kd)),
                                student=None, mode='query', c_num=len(specific))
                            query = self.transform_no_toTensor(query_tensor)
                            query_label_relative = torch.LongTensor(
                                (list(range(len(specific)))) * self.args.num_qur_kd).cuda(self.args.device)

                            task_data.append((support, support_label_relative, query, query_label_relative))
                        #     task_losses.append(self.tmeta_loss(teacher=teacher,support=support,support_label=support_label_relative,query=query,query_label=query_label_relative))
                        # meta_batch_loss = torch.mean(torch.stack(task_losses))
                if add_memory==True and tmeta_idicator == True and self.args.noteacher==False:
                    print('tmeta')
                    task_losses=[]
                    for support,support_label_relative,query,query_label_relative in task_data:
                        loss_outer = self.tmeta_loss(teacher=teacher, support=support,
                                                     support_label=support_label_relative, query=query,
                                                     query_label=query_label_relative)
                        task_losses.append(loss_outer)
                    meta_batch_loss = torch.mean(torch.stack(task_losses))
                    add_memory = False
                    tmeta_idicator = False

                else:# memory
                    memory_losses = []
                    for _ in range(self.args.episode_batch):
                        pos=np.random.uniform(0, 1)
                        # pos = 0
                        if pos>=0.5:
                            support_data, support_label_abs, query_data, query_label_abs, specific = self.synthesizer.get_random_task(num_w=self.args.way_train, num_s=self.args.num_sup_train, num_q=self.args.num_qur_train,mode='split')
                            support_label_relative = label_abs2relative(specific, support_label_abs).cuda(self.args.device)
                            query_label_relative = label_abs2relative(specific, query_label_abs).cuda(self.args.device)
                            support, support_label_relative, query, query_label_relative = support_data.cuda(self.args.device), support_label_relative.cuda(self.args.device), query_data.cuda(self.args.device), query_label_relative.cuda(self.args.device)

                            if self.args.teacherMethod!='protonet':
                                support, support_label_relative, query, query_label_relative = shuffle_task(self.args,support, support_label_relative, query, query_label_relative )

                            # visualize_tsne(self.model,support,support_label_relative, query, query_label_relative,'combination{}'.format(it_id))
                            memory_losses.append(self.meta_loss( support=support,support_label=support_label_relative, query=query,query_label=query_label_relative))
                        else:
                            dist = Beta(torch.FloatTensor([0.5]), torch.FloatTensor([0.5]))
                            x1s, y1s,x1q,y1q,specific=self.synthesizer.get_random_task(num_w=self.args.way_train, num_s=self.args.num_sup_train, num_q=self.args.num_qur_train,mode='split')
                            x2s, y2s,x2q,y2q,_=self.synthesizer.get_random_task(num_w=self.args.way_train, num_s=self.args.num_sup_train, num_q=self.args.num_qur_train,mode='split')
                            support_label_relative = label_abs2relative(specific, y1s).cuda(self.args.device)
                            query_label_relative = label_abs2relative(specific, y1q).cuda(self.args.device)
                            x1s, y1s,x1q,y1q=x1s.cuda(self.args.device), y1s.cuda(self.args.device),x1q.cuda(self.args.device),y1q.cuda(self.args.device)
                            x2s, y2s,x2q,y2q=x2s.cuda(self.args.device), y2s.cuda(self.args.device),x2q.cuda(self.args.device),y2q.cuda(self.args.device)
                            lam_mix = dist.sample().to(self.args.device)
                            x_mix_s, _ = mixup_data(self.model.getFeature(x1s), self.model.getFeature(x2s), lam_mix)
                            x_mix_q, _ = mixup_data(self.model.getFeature(x1q), self.model.getFeature(x2q), lam_mix)
                            # visualize_tsne(self.model, x_mix_s, support_label_relative, x_mix_q, query_label_relative,'mixup{}'.format(it_id),True)
                            z_support = x_mix_s.contiguous().view(self.args.way_train * self.args.num_sup_train, -1)
                            z_query = x_mix_q.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)
                            protos = []
                            for c in range(self.args.way_train):
                                protos.append(z_support[support_label_relative == c].mean(0))
                            z_proto = torch.stack(protos, dim=0)
                            z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)
                            dists = euclidean_dist(z_query, z_proto)
                            score = -dists
                            loss_outer = self.loss_fn(score, query_label_relative)
                            memory_losses.append(loss_outer)
                    meta_batch_loss = torch.mean(torch.stack(memory_losses))
                    add_memory = False
                    tmeta_idicator = False

                writer.add_scalar(tag='train_loss', scalar_value=meta_batch_loss.item(),global_step=it_id)
                self.optimizer.zero_grad()
                meta_batch_loss.backward()
                self.optimizer.step()

                if self.args.earlyStop == 0:
                    if (it_id+1)%self.args.generate_interval==0:
                        tmeta_idicator=True
                        add_memory=True
                if self.args.earlyStop==1:
                    raise NotImplementedError
                if self.args.earlyStop == 2:
                    # early_stop according to val_acc
                    val_acc_all = []
                    for val_id, val_batch in enumerate(self.val_loader):
                        data, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(self.args.device)
                        support, support_label_relative, query, query_label_relative = data2supportquery(self.args,
                                                                                                         data)
                        val_acc = self.test_once(support=support, support_label_relative=support_label_relative,
                                                 query=query, query_label_relative=query_label_relative)
                        val_acc_all.append(val_acc)
                        if val_id == 3:
                            break
                    writer.add_scalar(tag='val_acc', scalar_value=torch.stack(val_acc_all).mean().item(),
                                      global_step=it_id)
                    if e_maxValAcc == None or torch.stack(val_acc_all).mean() > e_maxValAcc:
                        e_maxValAcc = torch.stack(val_acc_all).mean()
                        e_count = 0
                    else:
                        e_count = e_count + 1
                        if e_count == self.args.generate_interval:
                            print('generate after', it_id)
                            tmeta_idicator = True
                            add_memory = True
                            e_count = 0
                            e_maxValAcc = None




                if self.args.rewardType==0:
                    # long reward 0
                    if self.args.scheduleScheduler:
                        # val & optimize shceduler
                        val_acc_all = []
                        for val_id,val_batch in enumerate(self.val_loader):
                            data, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(self.args.device)
                            support, support_label_relative, query, query_label_relative = data2supportquery(self.args,
                                                                                                             data)
                            val_acc = self.test_once(support=support, support_label_relative=support_label_relative,
                                                     query=query, query_label_relative=query_label_relative)
                            val_acc_all.append(val_acc)
                            if val_id==2:
                                break
                        # for val_id in range(2):
                        #     support, support_label_relative, query, query_label_relative = get_random_task_from_dataset(args=self.args,data=val_data,label_dataset=val_label_dataset,num_class=valset.num_class)
                        #     support, support_label_relative, query, query_label_relative=support.cuda(self.args.device), support_label_relative.cuda(self.args.device), query.cuda(self.args.device), query_label_relative.cuda(self.args.device)
                        #     val_acc = self.test_once(support=support, support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
                        #     val_acc_all.append(val_acc)
                        reward = torch.stack(val_acc_all).mean()
                        writer.add_scalar(tag='val_acc', scalar_value=reward.item(), global_step=it_id)
                        loss_scheduler = 0
                        if add_memory==False:
                            all_task_weight = self.scheduler.forward(self.scheduler.resource[0], self.scheduler.resource[1],
                                                                     self.scheduler.resource[2], self.scheduler.resource[3])
                            all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)
                            self.scheduler.m = Categorical(all_task_prob)
                        for s_i in selected_tasks_idx:
                            loss_scheduler = loss_scheduler - self.scheduler.m.log_prob(s_i)
                        loss_scheduler *= (reward - moving_avg_reward)
                        moving_avg_reward = self.update_moving_avg(moving_avg_reward, reward, schedule_id)
                        self.scheduler_optimizer.zero_grad()
                        loss_scheduler.backward()
                        self.scheduler_optimizer.step()
                        schedule_id = schedule_id + 1
                    elif self.args.scheduleWeight:
                        # val & optimize shceduler
                        if self.args.earlyStop!=2:
                            val_acc_all = []
                            for val_id, val_batch in enumerate(self.val_loader):
                                data, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(self.args.device)
                                support, support_label_relative, query, query_label_relative = data2supportquery(self.args,data)
                                val_acc = self.test_once(support=support, support_label_relative=support_label_relative,
                                                         query=query, query_label_relative=query_label_relative)
                                val_acc_all.append(val_acc)
                                if val_id == 3:
                                    break
                            # for val_id in range(2):
                            #     support, support_label_relative, query, query_label_relative = get_random_task_from_dataset(args=self.args,data=val_data,label_dataset=val_label_dataset,num_class=valset.num_class)
                            #     support, support_label_relative, query, query_label_relative=support.cuda(self.args.device), support_label_relative.cuda(self.args.device), query.cuda(self.args.device), query_label_relative.cuda(self.args.device)
                            #     val_acc = self.test_once(support=support, support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
                            #     val_acc_all.append(val_acc)
                        else:
                            pass
                        reward = torch.stack(val_acc_all).mean()
                        writer.add_scalar(tag='val_acc', scalar_value=reward.item(), global_step=it_id)
                        loss_scheduler = 0
                        candidate_weight = self.model_pool_weight[candidate_ids]
                        candidate_prob = torch.softmax(candidate_weight.reshape(-1), dim=-1)
                        m = Categorical(candidate_prob)
                        for s_i in selected_tasks_idx:
                            loss_scheduler = loss_scheduler - m.log_prob(torch.tensor(s_i).cuda(self.args.device))
                        loss_scheduler *= (reward - moving_avg_reward)
                        moving_avg_reward = self.update_moving_avg(moving_avg_reward, reward, schedule_id)
                        self.scheduler_optimizer.zero_grad()
                        loss_scheduler.backward()
                        self.scheduler_optimizer.step()
                        schedule_id = schedule_id + 1
                        self.calculate_weight(it_id=it_id, writer=writer)

                elif self.args.rewardType==1:
                    if self.args.scheduleWeight:
                        if self.args.earlyStop != 2:
                            val_acc_all = []
                            for val_id, val_batch in enumerate(self.val_loader):
                                data, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(self.args.device)
                                support, support_label_relative, query, query_label_relative = data2supportquery(self.args,
                                                                                                                 data)
                                val_acc = self.test_once(support=support, support_label_relative=support_label_relative,
                                                         query=query, query_label_relative=query_label_relative)
                                val_acc_all.append(val_acc)
                                if val_id == 3:
                                    break
                        reward = torch.stack(val_acc_all).mean()
                        writer.add_scalar(tag='val_acc', scalar_value=reward.item(), global_step=it_id)
                        self.long_reward.append(reward)
                        if ((it_id) % 20) == 0:
                            reward=torch.stack(self.long_reward).mean()
                            loss_scheduler = 0
                            candidate_weight = self.model_pool_weight[candidate_ids]
                            candidate_prob = torch.softmax(candidate_weight.reshape(-1), dim=-1)
                            m = Categorical(candidate_prob)
                            for s_i in selected_tasks_idx:
                                loss_scheduler = loss_scheduler - m.log_prob(torch.tensor(s_i).cuda(self.args.device))
                            loss_scheduler *= (reward - moving_avg_reward)
                            moving_avg_reward = self.update_moving_avg(moving_avg_reward, reward, schedule_id)
                            self.scheduler_optimizer.zero_grad()
                            loss_scheduler.backward()
                            self.scheduler_optimizer.step()
                            schedule_id = schedule_id + 1
                            self.calculate_weight(it_id=it_id, writer=writer)
                            self.long_reward=[]

                    elif self.args.scheduleScheduler:
                        raise NotImplementedError
                elif self.args.rewardType == 2:
                    if self.args.scheduleWeight:
                        if self.args.earlyStop != 2:
                            val_acc_all = []
                            for val_id, val_batch in enumerate(self.val_loader):
                                data, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(self.args.device)
                                support, support_label_relative, query, query_label_relative = data2supportquery(
                                    self.args,
                                    data)
                                val_acc = self.test_once(support=support,
                                                         support_label_relative=support_label_relative,
                                                         query=query, query_label_relative=query_label_relative)
                                val_acc_all.append(val_acc)
                                if val_id == 3:
                                    break
                        reward = torch.stack(val_acc_all).mean()
                        writer.add_scalar(tag='val_acc', scalar_value=reward.item(), global_step=it_id)
                        self.sub_long_reward.append(reward)
                        self.long_reward.append(reward)
                        #if ((it_id) % 20) == 0:
                        if ((it_id) % 4) == 0 or add_memory==True:
                            reward=torch.stack(self.sub_long_reward).mean()
                            loss_scheduler = 0
                            candidate_weight = self.model_pool_weight[candidate_ids]
                            candidate_prob = torch.softmax(candidate_weight.reshape(-1), dim=-1)
                            m = Categorical(candidate_prob)
                            for s_i in selected_tasks_idx:
                                loss_scheduler = loss_scheduler - m.log_prob(
                                    torch.tensor(s_i).cuda(self.args.device))
                            loss_scheduler *= (reward - moving_avg_reward)
                            self.scheduler_optimizer.zero_grad()
                            loss_scheduler.backward()
                            self.scheduler_optimizer.step()
                            self.calculate_weight(it_id=it_id, writer=writer)
                            self.sub_long_reward=[]
                        if add_memory == True:
                            moving_avg_reward = self.update_moving_avg(moving_avg_reward,
                                                                       torch.stack(self.long_reward).mean(),
                                                                       schedule_id)
                            schedule_id = schedule_id + 1
                            self.long_reward = []









                # test
                if it_id % 200 == 0:
                    timer_test = Timer()
                    test_acc_avg,test_pm=self.test_loop()
                    writer.add_scalar(tag='test_acc', scalar_value=test_acc_avg.item(),global_step=it_id)
                    if test_acc_avg > test_acc_max:
                        test_acc_max = test_acc_avg
                        max_it = it_id
                        max_pm = test_pm
                        torch.save(self.model.state_dict(), self.checkpoints_path + '/bestmodel_{}it.pth'.format(max_it))
                    self.logger.info('[Epoch]:{}, [TestAcc]:{} +- {}. [BestEpoch]:{}, [BestTestAcc]:{} +- {}.'.format(
                        it_id, test_acc_avg, test_pm, max_it, test_acc_max, max_pm))
                    print(timer_test.measure())

                if it_id % 200 == 0:
                    print('ETA:{}/{}'.format(
                        timer.measure(),
                        timer.measure((it_id) / (self.args.episode_train//self.args.episode_batch)))
                    )


    def test_once(self,support,support_label_relative,query, query_label_relative):
        if self.args.teacherMethod=='maml':
            self.model.zero_grad()
            fast_parameters = list(self.model.parameters())
            for weight in self.model.parameters():
                weight.fast = None

            for task_step in range(self.args.test_inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label_relative)
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
                if self.args.approx:
                    grad = [g.detach() for g in grad]
                fast_parameters = []
                for k, weight in enumerate(self.model.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[ k]
                    fast_parameters.append( weight.fast)
            # outer
            correct=0
            total=0
            scores = self.forward(query)
            prediction = torch.max(scores, 1)[1]
            correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
            total = total + len(query_label_relative)
            acc=1.0*correct/total*100.0

            self.model.zero_grad()
            for weight in self.model.parameters():
                weight.fast = None
        elif self.args.teacherMethod=='protonet':
            self.model.zero_grad()
            for weight in self.model.parameters():
                weight.fast = None
            z_support = self.model.getFeature(support)
            z_query = self.model.getFeature(query)
            z_support = z_support.contiguous().view(self.args.way_train * self.args.num_sup_train, -1)
            z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)
            z_support = z_support.contiguous()
            protos = []
            for c in range(self.args.way_train):
                protos.append(z_support[support_label_relative == c].mean(0))
            z_proto = torch.stack(protos, dim=0)
            z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)
            dists = euclidean_dist(z_query, z_proto)
            score = -dists
            logprobs = score
            correct, total = 0, 0
            prediction = torch.max(logprobs, 1)[1]
            correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
            total = total + len(query_label_relative)
            acc = 1.0 * correct / total * 100.0
            self.model.zero_grad()
            for weight in self.model.parameters():
                weight.fast = None
        elif self.args.teacherMethod=='anil':
            self.model.zero_grad()
            fast_parameters = list(self.model.trunk[-1].parameters())
            for weight in self.model.parameters():
                weight.fast = None

            for task_step in range(self.args.test_inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label_relative)
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)
                if self.args.approx:
                    grad = [g.detach() for g in grad]
                fast_parameters = []
                for k, weight in enumerate(self.model.trunk[-1].parameters()):
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[k]
                    fast_parameters.append(weight.fast)
                    # outer
            correct = 0
            total = 0
            scores = self.forward(query)
            prediction = torch.max(scores, 1)[1]
            correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
            total = total + len(query_label_relative)
            acc = 1.0 * correct / total * 100.0

            self.model.zero_grad()
            for weight in self.model.parameters():
                weight.fast = None
        else:
            raise NotImplementedError
        return acc
    def test_loop(self):
        test_acc_all = []
        for test_batch in self.test_loader:
            data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc=self.test_once(support=support,support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
            test_acc_all.append(test_acc)
        test_acc_avg, pm = compute_confidence_interval(test_acc_all)
        return test_acc_avg,pm

    def get_differential_parameter_dict(self,params):
        params=OrderedDict(params)
        length=len(params)
        param_dict = dict()
        indexes = []
        for i, (name, param) in enumerate(params.items()):
            if self.args.teacherMethod=='protonet':
                if i==length-2:
                    break
            param_dict[name] = param.cuda(self.args.device)
            indexes.append(i)
        #print(indexes)
        return param_dict, indexes

    def update_moving_avg(self,mv, reward, count):
        return mv + (reward.item() - mv) / (count + 1)

    def calculate_weight(self,it_id,writer):
        if self.args.scheduleWeight:
            all_weight=torch.softmax(self.model_pool_weight.reshape(-1),dim=-1)
            avg_weight_normal=torch.sum(all_weight[torch.tensor(self.model_pool_attribute)==0])*100.0
            avg_weight_ood=torch.sum(all_weight[torch.tensor(self.model_pool_attribute)==1])*100.0
            avg_weight_fake=torch.sum(all_weight[torch.tensor(self.model_pool_attribute)==2])*100.0
            self.logger.info('[Epoch]:{}, [weight_normal]:{}. [weight_ood]:{}. [weight_fake]:{}.'.format(
                it_id, avg_weight_normal, avg_weight_ood, avg_weight_fake))
            writer.add_scalar(tag='avg_weight_normal', scalar_value=avg_weight_normal, global_step=it_id)
            writer.add_scalar(tag='avg_weight_ood', scalar_value=avg_weight_ood, global_step=it_id)
            writer.add_scalar(tag='avg_weight_fake', scalar_value=avg_weight_fake, global_step=it_id)
        elif self.args.scheduleScheduler:
            pass
        else:
            writer.add_scalar(tag='avg_weight_normal', scalar_value=(1-self.args.oodP-self.args.fakeP)*100, global_step=it_id)
            writer.add_scalar(tag='avg_weight_ood', scalar_value=self.args.oodP*100, global_step=it_id)
            writer.add_scalar(tag='avg_weight_fake', scalar_value=self.args.fakeP*100, global_step=it_id)
def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
def get_random_task_from_data(args,support_data,support_data_label_relative,query_data,query_data_label_relative):
    support,support_label_relative,query,query_label_relative=[],[],[],[]
    for c in range(args.way_train):
        select=support_data[support_data_label_relative==c]
        support_select=torch.stack(random.sample(list(select),args.num_sup_train),dim=0)
        support.append(support_select[:args.num_sup_train])
        support_label_relative.append(torch.LongTensor([c] * args.num_sup_train))

        select = query_data[query_data_label_relative == c]
        query_select = torch.stack(random.sample(list(select), args.num_qur_train), dim=0)
        query.append(query_select[:args.num_qur_train])
        query_label_relative.append(torch.LongTensor([c] * args.num_qur_train))
    support=torch.cat(support,dim=0)
    query=torch.cat(query,dim=0)
    support_label_relative=torch.cat(support_label_relative,dim=0)
    query_label_relative = torch.cat(query_label_relative, dim=0)
    return support,support_label_relative.cuda(args.device),query,query_label_relative.cuda(args.device)

# def get_random_task_from_model_pool_data(args,model_pool_data,model_pool_specific):
#     specific=[]
#     data_memory=[]
#     label_memory=[]
#     while True:
#         if len(specific)==args.way_train:
#             break
#         model_location=random.randint(0,len(model_pool_data)-1)
#         label_location=random.randint(0,len(model_pool_specific[model_location])-1)
#         label_abs=model_pool_specific[model_location][label_location]
#         if label_abs in specific:
#             continue
#         else:
#             specific.append(label_abs)
#             data,label_relative=model_pool_data[model_location]
#             data_memory.append(data[label_relative==label_location])
#             label_memory.append(torch.LongTensor([len(specific)-1]*(sum(label_relative==label_location))).cuda(args.device))
#     data_memory=torch.cat(data_memory,dim=0)
#     label_memory=torch.cat(label_memory,dim=0)
#     support, support_label_relative, query, query_label_relative=get_random_task_from_data(args,data_memory,label_memory)
#     return support, support_label_relative, query, query_label_relative

def get_random_task_from_dataset(args,data,label_dataset,num_class):
    select_way=random.sample(list(range(num_class)),args.way_train)
    support,support_label_relative,query,query_label_relative=[],[],[],[]
    for c,c_dataset in enumerate(select_way):
        select=data[label_dataset==c_dataset]
        support_query_select=torch.stack(random.sample(list(select),args.num_sup_train+args.num_qur_train),dim=0)
        support.append(support_query_select[:args.num_sup_train])
        support_label_relative.append(torch.LongTensor([c] * args.num_sup_train))

        query.append(support_query_select[args.num_sup_train:args.num_sup_train+args.num_qur_train])
        query_label_relative.append(torch.LongTensor([c] * args.num_qur_train))
    support=torch.cat(support,dim=0)
    query=torch.cat(query,dim=0)
    support_label_relative=torch.cat(support_label_relative,dim=0)
    query_label_relative = torch.cat(query_label_relative, dim=0)
    return support,support_label_relative,query,query_label_relative

def get_random_task_from_data2(args,data,label_relative):
    support,support_label_relative,query,query_label_relative=[],[],[],[]
    for c in range(args.way_train):
        select=data[label_relative==c]
        support_query=torch.stack(random.sample(list(select),args.num_sup_train+args.num_qur_train),dim=0)
        support.append(support_query[:args.num_sup_train])
        query.append(support_query[args.num_sup_train:args.num_sup_train+args.num_qur_train])
        support_label_relative.append(torch.LongTensor([c] * args.num_sup_train))
        query_label_relative.append(torch.LongTensor([c] * args.num_qur_train))
    support=torch.cat(support,dim=0)
    query=torch.cat(query,dim=0)
    support_label_relative=torch.cat(support_label_relative,dim=0)
    query_label_relative = torch.cat(query_label_relative, dim=0)
    return support,support_label_relative.cuda(args.device),query,query_label_relative.cuda(args.device)

def visualize_tsne(model, x1,y1,x2,y2,name,feature_input=False):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    tsne = TSNE(n_components=2, random_state=42,perplexity=10)
    x=torch.cat((x1,x2),dim=0)
    y=torch.cat((y1,y2),dim=0)
    if feature_input==False:
        with torch.no_grad():
            features=model.getFeature(x)
            features=features.cpu().numpy()
            y=y1.cpu().numpy()
    else:
        features=x
        features = features.detach().cpu().numpy()
        y = y1.cpu().numpy()
    transformed = tsne.fit_transform(features)
    # 可视化
    plt.figure(figsize=(10, 10))
    for i,label in enumerate(set(y)):
        circle=10
        idx = np.where(np.array(y) == label)[0]
        color = colors[i%circle]
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=str(label), color=color, edgecolors='w', alpha=0.9, linewidth=0.9,s=50)

    plt.savefig('./tsne/{}.png'.format(name))