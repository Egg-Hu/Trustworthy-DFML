
import os

import numpy as np

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
    label_abs2relative
from generator import Generator,GeneratorCon
from synthesis import Synthesizer, DeepInvSyntheiszer
from synthesis.concontrastive import ConSynthesizer
from tensorboardX import SummaryWriter
from black_box_tool import set_maml, get_model, get_random_teacher, kldiv, data2supportquery, \
    compute_confidence_interval, get_dataloader, Timer,shuffle_task, bias , construct_model_pool,dataset_classnum
from logger import get_logger
from synthesis._utils import save_image_batch
from torch.distributions.categorical import Categorical



class PRE(nn.Module):
    def __init__(self,args):
        super(PRE, self).__init__()
        self.args=args
        # file
        feature1 = '{}_{}_{}'.format(args.method, args.teacherMethod, args.inversionMethod)
        if self.args.dataset==self.args.testdataset:
            feature2_1 = '{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset,args.pre_backbone,args.backbone, args.APInum, args.oodP,args.fakeP, args.fakefrom)
        else:
            feature2_1 = '{}/{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset, args.testdataset,args.pre_backbone,args.backbone, args.APInum, args.oodP,args.fakeP, args.fakefrom)
        if args.teacherMethod == 'maml':
            feature2_2 = '{}wPre_{}S_{}Q_{}kds_{}kdq_{}stepkd_{}steptrain_{}teststep_{}innerlr_{}outerlr_{}batch_{}Git_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.num_sup_kd, args.num_qur_kd,
                args.inner_update_num_kd, args.inner_update_num,
                args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,
                 args.generate_iterations, args.Glr)
        elif args.teacherMethod == 'protonet':
            feature2_2 = '{}wPre_{}S_{}Q_{}lr_{}batch_{}InvM_{}Ginterval_{}Git_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.outer_lr, args.episode_batch,
                args.inversionMethod,  args.generate_interval, args.generate_iterations, args.Glr)
        elif args.teacherMethod == 'anil':
            feature2_2 = '{}wPre_{}S_{}Q_{}kds_{}kdq_{}stepkd_{}steptrain_{}teststep_{}innerlr_{}outerlr_{}batch_{}Git_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.num_sup_kd, args.num_qur_kd,
                args.inner_update_num_kd, args.inner_update_num,
                args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,
                args.generate_iterations, args.Glr)
        feature2 = feature2_1 + '_' + feature2_2
        if args.approx:
            feature2 = feature2 + '_1Order'
        if (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==False and self.args.scheduleWeight==False):
            self.checkpoints_path = './checkpoints_final/'+'ATTACK/'+ feature1 + '/' + feature2
        elif (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==True or self.args.scheduleWeight==True):
            self.checkpoints_path = './checkpoints_final/' + 'SCHEDULE/' + feature1 + '/' + feature2
        elif self.args.oodP==0 or self.args.fakeP==0:
            self.checkpoints_path = './checkpoints_final/' + 'SAFE/' + feature1 + '/' + feature2
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
        self.transform_no_toTensor = get_transform_no_toTensor(args)
        self.transform = get_transform(args)
        if (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==False and self.args.scheduleWeight==False):
            datapool_path = self.datapool_path = './datapool_final/' + 'ATTACK/' + os.path.join(feature1, feature2)
        elif (self.args.oodP!=0 or self.args.fakeP!=0) and (self.args.scheduleScheduler ==True or self.args.scheduleWeight==True):
            datapool_path = self.datapool_path = './datapool_final/' + 'SCHEDULE/' + os.path.join(feature1, feature2)
        elif self.args.oodP==0 or self.args.fakeP==0:
            datapool_path = self.datapool_path = './datapool_final/' + 'SAFE/' + os.path.join(feature1, feature2)

        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print('remove')
        #os.makedirs('./datapool/' + os.path.join(feature1, feature2), exist_ok=True)

        if self.args.preGenerate:
            max_batch_per_class = 20
            self.preGenerate_path = './preGenerate/' + os.path.join(feature1, feature2)
        else:
            max_batch_per_class = 20
        if args.inversionMethod == 'cmi':
            raise NotImplementedError
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
            raise NotImplementedError
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
        model_pool_path,model_pool_attribute=construct_model_pool(args=self.args,unique=True)
        model_pool_specific=[]
        #prepare for model_pool_specific
        timer_model=Timer()
        for i,path in enumerate(model_pool_path):
            tmp = torch.load(path)
            specific = tmp['specific']
            specific = [i + bias[self.args.dataset] for i in specific]
            if model_pool_attribute[i]==1:
                specific = random.sample(range(dataset_classnum[self.args.dataset]), self.args.way_pretrain)
                specific = [s + bias[self.args.dataset] for s in specific]
            model_pool_specific.append(specific)
        print('time for model:',timer_model.measure())

        #prepare for model_pool_data
        print('data pregenerate start!')
        for t_id in range(len(model_pool_path)):
            tmp = torch.load(model_pool_path[t_id])
            teacher = get_model(args=self.args, type=self.args.pre_backbone, set_maml_value=False,
                                arbitrary_input=False).cuda(self.args.device)
            teacher.load_state_dict(tmp['teacher'])
            specific = model_pool_specific[t_id]
            teacher.eval()
            self.synthesizer.teacher = teacher
            self.synthesizer.c_abs_list = specific
            _ = self.synthesizer.synthesize(
                targets=torch.LongTensor((list(range(len(specific)))) * (self.args.num_sup_train+self.args.num_qur_train)), student=None,
                mode='all', c_num=len(specific))
        print('data pregenerate end!')

        timer = Timer()
        with SummaryWriter(self.writer_path) as writer:
            test_acc_max = 0
            max_it = 0
            max_pm = 0
            for it_id in range(1, self.args.episode_train//self.args.episode_batch + 1):
                memory_losses = []
                for _ in range(self.args.episode_batch):
                    support_data, support_label_abs, query_data, query_label_abs, specific = self.synthesizer.get_random_task(num_w=self.args.way_train, num_s=self.args.num_sup_train, num_q=self.args.num_qur_train,mode='all')
                    support_label_relative = label_abs2relative(specific, support_label_abs).cuda(self.args.device)
                    query_label_relative = label_abs2relative(specific, query_label_abs).cuda(self.args.device)
                    support, support_label_relative, query, query_label_relative = support_data.cuda(self.args.device), support_label_relative.cuda(self.args.device), query_data.cuda(self.args.device), query_label_relative.cuda(self.args.device)
                    if self.args.teacherMethod!='protonet':
                        support, support_label_relative, query, query_label_relative = shuffle_task(self.args,support, support_label_relative, query, query_label_relative )
                    memory_losses.append(self.meta_loss( support=support,support_label=support_label_relative, query=query,query_label=query_label_relative))
                meta_batch_loss = torch.mean(torch.stack(memory_losses))

                writer.add_scalar(tag='train_loss', scalar_value=meta_batch_loss.item(),global_step=it_id)
                self.optimizer.zero_grad()
                meta_batch_loss.backward()
                self.optimizer.step()

                # test
                if it_id % 500 == 0:
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
