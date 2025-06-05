import copy
import os
from collections import OrderedDict

import numpy as np
from torch.distributions import Categorical

import random
import shutil
import sys



sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch.nn.functional as F

import torch
import torch.nn as nn
from scheduler import Scheduler
from black_box_tool import get_transform_no_toTensor, get_transform, Normalizer, NORMALIZE_DICT, \
    label_abs2relative,dataset_classnum,construct_model_pool
from generator import Generator,GeneratorCon
from synthesis import Synthesizer, DeepInvSyntheiszer
from synthesis.concontrastive import ConSynthesizer
from tensorboardX import SummaryWriter
from black_box_tool import set_maml, get_model, get_random_teacher, kldiv, data2supportquery, bias,\
    compute_confidence_interval, get_dataloader, Timer,shuffle_task
from logger import get_logger
from synthesis._utils import save_image_batch


class PURER(nn.Module):
    def __init__(self,args):
        super(PURER, self).__init__()
        self.args=args
        #file
        # assert args.Glr==0.25, 'Glr'
        feature1='{}_{}_{}'.format(args.method,args.teacherMethod,args.inversionMethod)
        feature2_1 = '{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset, args.pre_backbone, args.backbone,args.APInum, args.oodP,args.fakeP, args.fakefrom)
        if self.args.teacherMethod=='maml':
            feature2_2 = 'MAML_{}wPre_{}S_{}Q_{}trainStep_{}testStep_{}innerlr_{}outerlr_{}TaskBatch_{}CandiBatch_{}Glr'.format(
                args.way_pretrain,args.num_sup_train,args.num_qur_train, args.inner_update_num,
                args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,args.candidate_size,args.Glr)
        elif self.args.teacherMethod=='protonet':
            feature2_2 = 'ProtoNet_{}wPre_{}S_{}Q_{}lr_{}TaskBatch_{}CandiBatch_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.outer_lr, args.episode_batch,args.candidate_size, args.Glr)
        elif self.args.teacherMethod=='anil':
            feature2_2 = 'ANIL_{}wPre_{}S_{}Q_{}trainStep{}testStep{}innerlr_{}outerlr_{}Taskbatch_{}CandiBatch_{}Glr'.format(
                args.way_pretrain, args.num_sup_train, args.num_qur_train, args.inner_update_num,
                args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,args.candidate_size, args.Glr)
        feature2=feature2_1+'_'+feature2_2

        if args.defensew:
            feature2 = feature2 + '_{}Weight'.format(args.Slr)
        elif args.defenses:
            feature2 = feature2 + '_{}Scheduler'.format(args.Slr)
        if args.defensew or args.defenses:
            feature2 = feature2 + '_{}WarmUp'.format(args.warmup)
        if args.dynamicSlr:
            feature2 = feature2 + '_{}CosineAnnealingWarmRestarts'.format(5000)
        if args.approx:
            feature2 = feature2 + '_1Order'
        if args.unique:
            feature2 = feature2 + '_Unique'
        if args.dynamicSpecific:
            feature2 = feature2 + '_DynamicSpecific'
        feature2 = feature2 + '_{}'.format(args.extra)
        if (self.args.oodP != 0 or self.args.fakeP != 0) and (self.args.defensew == False and self.args.defenses == False):
            self.checkpoints_path = './checkpoints6/'+'ATTACK/' + feature1 + '/' + feature2
        elif (self.args.oodP != 0 or self.args.fakeP != 0) and (self.args.defensew == True or self.args.defenses == True):
            self.checkpoints_path = './checkpoints6/' + 'DEFENSE/' + feature1 + '/' + feature2
        else:
            self.checkpoints_path = './checkpoints6/' + 'SAFE/' + feature1 + '/' + feature2
        self.writer_path = os.path.join(self.checkpoints_path, 'writer')
        if os.path.exists(self.writer_path):
            shutil.rmtree(self.writer_path)
        os.makedirs(self.writer_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.logger = get_logger(feature1 + '/' + feature2, output=self.checkpoints_path + '/' + 'log.txt')
        _, self.val_loader, self.test_loader = get_dataloader(self.args)
        self.feature1=feature1
        self.feature2=feature2
        #meta model
        set_maml(True)
        self.model=get_model(args=args,type=args.backbone,set_maml_value=True,arbitrary_input=False).cuda(self.args.device)
        set_maml(False)
        self.model.trunk[-1].bias.data.fill_(0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.outer_lr)
        self.loss_fn = nn.CrossEntropyLoss()

        #synthesizer
        self.allData = torch.randn([(args.num_sup_train+args.num_qur_train) * dataset_classnum[self.args.dataset], args.channel, args.img_size, args.img_size], requires_grad=True, device='cuda',dtype=torch.float)
        self.allData_label = []
        for c_abs in range(dataset_classnum[self.args.dataset]):
            self.allData_label.append(torch.LongTensor([c_abs+ bias[self.args.dataset]] * (args.num_sup_train+args.num_qur_train)))
        self.allData_label = torch.cat(self.allData_label, 0).cuda(self.args.device)
        self.optimizer_di = torch.optim.Adam([self.allData], lr=self.args.Glr)


        self.generate = True
        self.adv = False
        if self.args.unique==False:
            self.model_pool_path, self.model_pool_attribute = construct_model_pool(args=self.args, unique=False)
        else:
            self.model_pool_path, self.model_pool_attribute = construct_model_pool(args=self.args, unique=True)

        self.model_pool_specific = []
        for i, path in enumerate(self.model_pool_path):
            tmp = torch.load(path)
            specific = tmp['specific']
            specific = [i + bias[self.args.dataset] for i in specific]
            if self.model_pool_attribute[i] == 1:
                specific = random.sample(range(dataset_classnum[self.args.dataset]), self.args.way_pretrain)
                specific = [s + bias[self.args.dataset] for s in specific]
            self.model_pool_specific.append(specific)

        if self.args.defensew:
            assert self.args.Slr == 0.1
            self.model_pool_weight = torch.nn.Parameter(torch.ones(len(self.model_pool_path)).cuda(self.args.device)).requires_grad_()
            self.scheduler_optimizer = torch.optim.Adam([self.model_pool_weight], lr=self.args.Slr)
        elif self.args.defenses:
            assert self.args.Slr==0.001
            names_weights_copy, indexes = self.get_differential_parameter_dict(params=self.model.named_parameters())
            self.scheduler = Scheduler(args=self.args, N=len(names_weights_copy), grad_indexes=indexes,use_deepsets=True).cuda(self.args.device)
            self.scheduler_optimizer = torch.optim.Adam(self.scheduler.parameters(), lr=self.args.Slr)
            self.model_pool_data=self.prepare_model_pool_data()
        if self.args.defensew or self.args.defenses and self.args.dynamicSlr:
            #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.scheduler_optimizer,T_max=30000)
            self.slr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.scheduler_optimizer, T_0=5000, T_mult=1,eta_min=0.001)




        self.candidate_teacher_ids = None

    # def get_random_specific(self,id):
    #     if self.model_pool_attribute[id]==1 or self.model_pool_attribute[id]==2:
    #         specific = random.sample(range(dataset_classnum[self.args.dataset]), self.args.way_pretrain)
    #         specific = [s + bias[self.args.dataset] for s in specific]
    #     else:
    #         tmp = torch.load(self.model_pool_path[id])
    #         specific = tmp['specific']
    #         specific = [i + bias[self.args.dataset] for i in specific]
    #     return specific
    def get_teacher_from_pool(self,id):
        tmp = torch.load(self.model_pool_path[id])
        teacher = get_model(args=self.args, type=self.args.pre_backbone, set_maml_value=False,arbitrary_input=False).cuda(self.args.device)
        teacher.load_state_dict(tmp['teacher'])
        teacher.eval()
        return teacher
    def get_specific(self,id):
        if self.model_pool_attribute[id]==1 or self.model_pool_attribute[id]==2:
            specific = random.sample(range(dataset_classnum[self.args.dataset]), self.args.way_pretrain)
            specific = [s + bias[self.args.dataset] for s in specific]
        else:
            specific = self.model_pool_specific[id]
        return specific
    def prepare_model_pool_data(self):
        print('model pool data start!')
        if os.path.exists('./model_pool_data6/' + os.path.join(self.feature1, self.feature2)):
            self.model_pool_data = torch.load(
                './model_pool_data6/' + os.path.join(self.feature1, self.feature2) + '/model_pool_data.pth')
        else:
            self.transform_no_toTensor = get_transform_no_toTensor(self.args)
            generator = GeneratorCon(nz=256, ngf=64, img_size=32, nc=3).cuda(self.args.device)
            self.synthesizer = ConSynthesizer(self.args, None, None, generator, nz=256, num_classes=500,
                                              img_size=(3, 32, 32),
                                              iterations=200, lr_g=0.001,
                                              synthesis_batch_size=30,
                                              bn=1.0, oh=1.0, adv=0.0,
                                              save_dir='./datapool6/' + os.path.join(self.feature1, self.feature2), transform=get_transform(self.args),
                                              transform_no_toTensor=self.transform_no_toTensor,
                                              device=self.args.gpu, c_abs_list=None, max_batch_per_class=1)
            # self.synthesizer = DeepInvSyntheiszer(self.args, None, None,
            #                                       img_size=(3, 32, 32),
            #                                       iterations=20000, lr_g=0.25,
            #                                       synthesis_batch_size=30,
            #                                       adv=0.0, bn=0.01, oh=1, tv=1e-4, l2=0.0,
            #                                       save_dir='./datapool6/' + os.path.join(self.feature1, self.feature2), transform=get_transform(self.args),
            #                                       normalizer=Normalizer(**NORMALIZE_DICT[self.args.dataset]),
            #                                       device=self.args.gpu,
            #                                       num_classes=500, c_abs_list=None,
            #                                       max_batch_per_class=1)
            self.model_pool_data = []
            for i, path in enumerate(self.model_pool_path):
                timer=Timer()
                teacher=self.get_teacher_from_pool(i)
                specific = self.model_pool_specific[i]
                self.synthesizer.teacher = teacher
                self.synthesizer.c_abs_list = specific

                support_query_list = []
                support_query_label_relative_list = []
                for g_it in range(1):
                    support_query_tensor = self.synthesizer.synthesize(
                        targets=torch.LongTensor((list(range(len(specific)))) * (30)), student=None,
                        mode='all', c_num=len(specific), add=False)
                    support_query = self.transform_no_toTensor(support_query_tensor)
                    support_query_label_relative = torch.LongTensor((list(range(len(specific)))) * (30)).cuda(
                        self.args.device)
                    support_query_list.append(support_query)
                    support_query_label_relative_list.append(support_query_label_relative)

                support_query = torch.cat(support_query_list, dim=0)
                support_query_label_relative = torch.cat(support_query_label_relative_list, dim=0)
                self.model_pool_data.append((support_query, support_query_label_relative))
                print(timer.measure())
            os.makedirs('./model_pool_data6/' + os.path.join(self.feature1, self.feature2), exist_ok=True)
            torch.save(self.model_pool_data,'./model_pool_data6/' + os.path.join(self.feature1, self.feature2) + '/model_pool_data.pth')
        print('model pool data end!')
        return self.model_pool_data

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

    def forward(self,x):
        scores  = self.model(x)
        return scores
    def train_once(self,return_acc=False):
        # generate
        if self.generate == True:
            loss_data = 0
            loss2 = 0
            for candidate_teacher_id in self.candidate_teacher_ids:
                #teacher=self.model_pool_teacher[node_id]
                teacher=self.get_teacher_from_pool(candidate_teacher_id)
                # specific=self.get_specific(candidate_teacher_id)
                if self.args.dynamicSpecific:
                    specific = self.get_specific(candidate_teacher_id)
                else:
                    specific=self.model_pool_specific[candidate_teacher_id]
                #teacher, specific, acc, teacher_dataset, model_id = get_random_teacher(args=self.args)
                mask=torch.tensor([False for _ in range(len(self.allData_label))]).cuda(self.args.device)
                targets_rel=[]
                for c_rel,c_abs in enumerate(specific):
                    mask=mask + (self.allData_label==c_abs)
                    targets_rel.append(torch.LongTensor([c_rel]*(self.args.num_sup_train+self.args.num_qur_train)))
                targets_rel=torch.cat(targets_rel,dim=0).cuda(self.args.device)
                inputs=self.allData[mask]
                if self.args.dataset == 'cifar100':
                    loss_data = loss_data + get_images_cifar(net=teacher, var_scale=0.0001,bn_reg_scale = 0.01,l2_coeff=0.00001,first_bn_multiplier=1,pre_inputs=inputs,targets=targets_rel)
                elif self.args.dataset == 'miniimagenet':
                    loss_data = loss_data + get_images_cifar(net=teacher, var_scale=0.0001,bn_reg_scale = 0.01,l2_coeff=0.00001,first_bn_multiplier=1,pre_inputs=inputs,targets=targets_rel)
                elif self.args.dataset == 'cub':
                    loss_data = loss_data + get_images_cifar(net=teacher, var_scale=0.0001,bn_reg_scale = 0.01,l2_coeff=0.00001,first_bn_multiplier=1,pre_inputs=inputs,targets=targets_rel)
                elif self.args.dataset == 'flower':
                    loss_data = loss_data + get_images_cifar(net=teacher, var_scale=0.0001,bn_reg_scale = 0.01,l2_coeff=0.00001,first_bn_multiplier=1,pre_inputs=inputs,targets=targets_rel)
                else:
                    raise NotImplementedError
                loss2 = loss2 + loss_data
            if self.adv:
                support_data, support_label_relative, query_data, query_label_relative = generate_random_task_data_fromDI(self.args,self.allData,self.allData_label)
                support, support_label_relative, query, query_label_relative = support_data.cuda(self.args.device), support_label_relative.cuda(self.args.device), query_data.cuda(self.args.device), query_label_relative.cuda(self.args.device)
                loss_outer=self.get_outer_loss(support=support,support_label_relative=support_label_relative,query=query,query_label_relative=query_label_relative)
                loss2 = loss2 + -1 * 10.0 * loss_outer
                self.adv=False
            self.optimizer_di.zero_grad()
            loss2.backward()
            self.optimizer_di.step()
            self.generate=False
        support_data, support_label_relative, query_data, query_label_relative = generate_random_task_data_fromDI(self.args, self.allData.detach(), self.allData_label)
        support, support_label_relative, query, query_label_relative = support_data.cuda(self.args.device), support_label_relative.cuda(self.args.device), query_data.cuda(self.args.device), query_label_relative.cuda(self.args.device)
        loss_outer,train_acc = self.get_outer_loss(support=support, support_label_relative=support_label_relative, query=query,query_label_relative=query_label_relative,return_acc=True)
        if return_acc:
            return loss_outer,train_acc
        else:
            return loss_outer
    def train_loop(self):
        timer = Timer()
        moving_avg_reward=40
        schedule_id=0
        with SummaryWriter(self.writer_path) as writer:
            test_acc_max = 0
            max_it = 0
            max_pm = 0
            train_acc_max=None
            self.generate=True
            self.adv = False
            for it_id in range(1,self.args.episode_train//self.args.episode_batch+1):
                loss_batch = []
                acc_batch = []
                if (self.args.defensew==False and self.args.defenses==False) or it_id<=self.args.warmup:
                    self.candidate_teacher_ids = random.sample(range(len(self.model_pool_path)),self.args.candidate_size)
                elif self.args.defensew:
                    with torch.no_grad():
                        candidate_weight = self.model_pool_weight[:]
                        candidate_prob = torch.softmax(candidate_weight.reshape(-1), dim=-1)
                        candidate_prob = candidate_prob.detach().cpu().numpy()
                        self.candidate_teacher_ids = np.random.choice(range(len(self.model_pool_path)),
                                                                      p=candidate_prob / np.sum(candidate_prob),
                                                                      size=self.args.candidate_size, replace=False)
                elif self.args.defenses:
                    subpool_ids,subpool_losses,task1_task2_losses,train_val_losses=self.prepare_scheduler()
                    _, _, all_task_weight = self.scheduler.get_weight(task_losses=subpool_losses,
                                                                      task1_task2_losses=task1_task2_losses,
                                                                      train_val_losses=train_val_losses,
                                                                      model=self.model, pt=int(it_id / (self.args.episode_train // self.args.episode_batch) * 100),
                                                                      model_ids=subpool_ids, detach=False)

                    torch.cuda.empty_cache()
                    all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)
                    self.select_teacher_ids=self.scheduler.sample_task(all_task_prob, self.args.candidate_size, replace=0)
                    self.candidate_teacher_ids=[]
                    for select_id in self.select_teacher_ids:
                        self.candidate_teacher_ids.append(subpool_ids[select_id])
                for _ in range(self.args.episode_batch):
                    if self.args.defensew==False and self.args.defenses==False:
                        self.candidate_teacher_ids = random.sample(range(len(self.model_pool_path)),self.args.candidate_size)
                    loss, train_acc = self.train_once(return_acc=True)
                    loss_batch.append(loss)
                    acc_batch.append(train_acc)


                self.generate = True
                loss = torch.stack(loss_batch).sum(0)
                train_acc = torch.stack(acc_batch).mean()
                writer.add_scalar(tag='train_loss', scalar_value=loss.item(),global_step=it_id)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if it_id <= 4000:
                    self.adv = False
                else:
                    if self.adv == False:
                        if train_acc_max == None or train_acc > train_acc_max:
                            train_acc_max = train_acc
                            e_count = 0
                        else:
                            e_count = e_count + 1
                            if e_count == 6:
                                self.adv = True
                                e_count = 0
                                train_acc_max = None

                #policy
                if (self.args.defensew or self.args.defensew) and it_id>self.args.warmup:
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
                    writer.add_scalar(tag='val_acc', scalar_value=reward.item(),global_step=it_id)
                    loss_scheduler = 0
                    if self.args.defensew:
                        candidate_weight = self.model_pool_weight[:]
                        candidate_prob = torch.softmax(candidate_weight.reshape(-1), dim=-1)
                        m = Categorical(candidate_prob)
                        for s_i in self.candidate_teacher_ids:
                            loss_scheduler = loss_scheduler - m.log_prob(torch.tensor(s_i).cuda(self.args.device))
                        loss_scheduler *= (reward - moving_avg_reward)
                        moving_avg_reward = self.update_moving_avg(moving_avg_reward, reward, schedule_id)
                        self.scheduler_optimizer.zero_grad()
                        loss_scheduler.backward()
                        self.scheduler_optimizer.step()
                        if self.args.defensew or self.args.defenses and self.args.dynamicSlr:
                            self.slr_scheduler.step()
                        schedule_id = schedule_id + 1
                    elif self.args.defenses:
                        for s_i in self.select_teacher_ids:
                            loss_scheduler = loss_scheduler - m.log_prob(s_i.cuda(self.args.device))
                        loss_scheduler *= (reward - moving_avg_reward)
                        moving_avg_reward = self.update_moving_avg(moving_avg_reward, reward, schedule_id)
                        self.scheduler_optimizer.zero_grad()
                        loss_scheduler.backward()
                        self.scheduler_optimizer.step()
                        if self.args.defensew or self.args.defenses and self.args.dynamicSlr:
                            self.slr_scheduler.step()
                        schedule_id = schedule_id + 1
                    self.candidate_teacher_ids = None

                # val
                if it_id%200 == 0 or it_id==1:
                    test_acc_avg, test_pm = self.test_loop()
                    writer.add_scalar(tag='test_acc', scalar_value=test_acc_avg.item(),
                                      global_step=it_id)
                    if test_acc_avg > test_acc_max:
                        test_acc_max = test_acc_avg
                        max_it = global_step=it_id
                        max_pm = test_pm
                        torch.save(self.model.state_dict(),
                                   self.checkpoints_path + '/bestmodel.pth')
                    self.logger.info(
                        '[Epoch]:{}, [TestAcc]:{} +- {}. [BestEpoch]:{}, [BestTestAcc]:{} +- {}.'.format(
                            it_id, test_acc_avg, test_pm, max_it, test_acc_max,
                            max_pm))

                    print('ETA:{}/{}'.format(
                        timer.measure(),
                        timer.measure(it_id / (self.args.episode_train//self.args.episode_batch)))
                    )
                    if self.args.defensew:
                        self.calculate_weight(it_id, writer)


    def calculate_weight(self,it_id,writer):
        all_weight=torch.softmax(self.model_pool_weight.reshape(-1),dim=-1)
        avg_weight_normal=torch.sum(all_weight[torch.tensor(self.model_pool_attribute)==0])*100.0
        avg_weight_ood=torch.sum(all_weight[torch.tensor(self.model_pool_attribute)==1])*100.0
        avg_weight_fake=torch.sum(all_weight[torch.tensor(self.model_pool_attribute)==2])*100.0
        self.logger.info('[Epoch]:{}, [weight_normal]:{}. [weight_ood]:{}. [weight_fake]:{}.'.format(
            it_id, avg_weight_normal, avg_weight_ood, avg_weight_fake))
        writer.add_scalar(tag='avg_weight_normal', scalar_value=avg_weight_normal, global_step=it_id)
        writer.add_scalar(tag='avg_weight_ood', scalar_value=avg_weight_ood, global_step=it_id)
        writer.add_scalar(tag='avg_weight_fake', scalar_value=avg_weight_fake, global_step=it_id)
    def prepare_scheduler(self):
        subpool_losses = []
        task1_task2_losses = []
        train_val_losses = []
        #subpool_ids = [id for id in range(len(self.model_pool_path))]
        subpool_ids = random.sample(list(range(len(self.model_pool_path))),int(self.candidate_size/self.args.oodP+self.args.fakeP))
        for model_id in subpool_ids:
            data, label_relative = self.model_pool_data[model_id][0], self.model_pool_data[model_id][1]
            support, support_label_relative, query, query_label_relative = get_random_task_from_data(args=self.args,data=data,label_relative=label_relative)

            subpool_losses.append(self.get_outer_loss(support=support, support_label_relative=support_label_relative, query=query,query_label_relative=query_label_relative).detach())

            support1, support_label_relative1, query1, query_label_relative1 = get_random_task_from_data(args=self.args,data=data,label_relative=label_relative)
            support2, support_label_relative2, query2, query_label_relative2 = get_random_task_from_data(args=self.args,data=data,label_relative=label_relative)
            support1, support_label_relative1, query1, query_label_relative1 = shuffle_task(args=self.args,
                                                                                            support=support1,
                                                                                            support_label=support_label_relative1,
                                                                                            query=query1,
                                                                                            query_label=query_label_relative1)
            support2, support_label_relative2, query2, query_label_relative2 = shuffle_task(args=self.args,
                                                                                            support=support2,
                                                                                            support_label=support_label_relative2,
                                                                                            query=query2,
                                                                                            query_label=query_label_relative2)
            task1_task2_losses.append((self.get_outer_loss(support=support1,
                                                      support_label_relative=support_label_relative1,
                                                      query=query1,
                                                      query_label_relative=query_label_relative1),
                                       self.get_outer_loss(support=support2,
                                                      support_label_relative=support_label_relative2,
                                                      query=query2,
                                                      query_label_relative=query_label_relative2)))
            for val_batch in self.val_loader:
                data_val, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(
                    self.args.device)
                support_val, support_label_relative_val, query_val, query_label_relative_val = data2supportquery(
                    self.args, data_val)
                break
            support_train, support_label_relative_train, query_train, query_label_relative_train = get_random_task_from_data(args=self.args, data=data, label_relative=label_relative)
            support_val, support_label_relative_val, query_val, query_label_relative_val = shuffle_task(
                args=self.args, support=support_val, support_label=support_label_relative_val,
                query=query_val,
                query_label=query_label_relative_val)
            support_train, support_label_relative_train, query_train, query_label_relative_train = shuffle_task(
                args=self.args, support=support_train, support_label=support_label_relative_train,
                query=query_train,
                query_label=query_label_relative_train)
            train_val_losses.append((self.get_outer_loss(support=support_train,
                                                    support_label_relative=support_label_relative_train,
                                                    query=query_train,
                                                    query_label_relative=query_label_relative_train),
                                     self.get_outer_loss(support=support_val,
                                                    support_label_relative=support_label_relative_val,
                                                    query=query_val,
                                                    query_label_relative=query_label_relative_val)))
        return subpool_ids,subpool_losses,task1_task2_losses,train_val_losses
    def test_once(self,support,support_label_relative,query, query_label_relative):
        if self.args.teacherMethod == 'maml':
            self.model.zero_grad()
            fast_parameters = list(
                self.model.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.model.parameters():
                weight.fast = None

            for task_step in range(self.args.test_inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label_relative)
                grad = torch.autograd.grad(loss_inner, fast_parameters, create_graph=True)  # build full graph support gradient of gradient
                if self.args.approx:
                    grad = [g.detach() for g in grad]  # do not calculate gradient of gradient if using first order approximation
                fast_parameters = []
                for k, weight in enumerate(self.model.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[ k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append( weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
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
        elif self.args.teacherMethod == 'protonet':
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
        elif self.args.teacherMethod == 'anil':
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
    def update_moving_avg(self,mv, reward, count):
        return mv + (reward.item() - mv) / (count + 1)
    def test_loop(self):
        test_acc_all = []
        for test_batch in self.test_loader:
            data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc=self.test_once(support=support,support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
            test_acc_all.append(test_acc)
        test_acc_avg, pm = compute_confidence_interval(test_acc_all)
        return test_acc_avg,pm

    def get_outer_loss(self, support, support_label_relative, query, query_label_relative,return_acc=False):
        if self.args.teacherMethod == 'maml':
            self.model.zero_grad()
            fast_parameters = list(self.model.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.model.parameters():
                weight.fast = None

            for task_step in range(self.args.inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label_relative)
                grad = torch.autograd.grad(loss_inner, fast_parameters,create_graph=True)  # build full graph support gradient of gradient
                if self.args.approx:
                    grad = [g.detach() for g in
                            grad]  # do not calculate gradient of gradient if using first order approximation
                fast_parameters = []
                for k, weight in enumerate(self.model.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append(weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
            # outer
            scores = self.forward(query)
            loss_outer = self.loss_fn(scores, query_label_relative)
            correct = 0
            total = 0
            prediction = torch.max(scores, 1)[1]
            correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
            total = total + len(query_label_relative)
            acc = 1.0 * correct / total * 100.0
            for weight in self.model.parameters():
                weight.fast = None
        elif self.args.teacherMethod == 'protonet':
            way=self.args.way_train
            z_support = self.model.getFeature(support)
            z_query = self.model.getFeature(query)
            z_support = z_support.contiguous().view(way * self.args.num_sup_train, -1)
            z_query = z_query.contiguous().view(way * self.args.num_qur_train, -1)
            z_support = z_support.contiguous()
            protos = []
            for c in range(way):
                protos.append(z_support[support_label_relative == c].mean(0))
            z_proto = torch.stack(protos, dim=0)
            z_query = z_query.contiguous().view(way * self.args.num_qur_train, -1)
            dists = euclidean_dist(z_query, z_proto)
            score = -dists
            loss_outer = self.loss_fn(score, query_label_relative)
            logprobs = score
            correct, total = 0, 0
            prediction = torch.max(logprobs, 1)[1]
            correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
            total = total + len(query_label_relative)
            acc = 1.0 * correct / total * 100.0
        elif self.args.teacherMethod == 'anil':
            # inner
            self.model.zero_grad()
            fast_parameters = list(self.model.trunk[
                                       -1].parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.model.parameters():
                weight.fast = None
            for task_step in range(self.args.inner_update_num):
                scores = self.forward(support)
                loss_inner = self.loss_fn(scores, support_label_relative)
                grad = torch.autograd.grad(loss_inner, fast_parameters,create_graph=True)  # build full graph support gradient of gradient
                if self.args.approx:
                    grad = [g.detach() for g in
                            grad]  # do not calculate gradient of gradient if using first order approximation
                fast_parameters = []
                for k, weight in enumerate(self.model.trunk[-1].parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.args.inner_lr * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - self.args.inner_lr * grad[
                            k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append(
                        weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
            # outer
            scores = self.forward(query)
            loss_outer = self.loss_fn(scores, query_label_relative)
            correct = 0
            total = 0
            prediction = torch.max(scores, 1)[1]
            correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
            total = total + len(query_label_relative)
            acc = 1.0 * correct / total * 100.0
            for weight in self.model.parameters():
                weight.fast = None
        if return_acc:
            return loss_outer,acc
        else:
            return loss_outer


def get_images_cifar(net,var_scale=0.001,bn_reg_scale = 10.0,l2_coeff=0.0,first_bn_multiplier=10,pre_inputs=None,targets=None):
    inputs=pre_inputs
    net.eval()
    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()
    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # setting up the range for jitter
    lim_0, lim_1 = 2, 2
    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))
    # foward with jit images
    net.zero_grad()
    outputs = net(inputs_jit)
    loss = criterion(outputs, targets)

    # apply total variation regularization
    diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
    diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
    diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
    diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
    loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss = loss + var_scale*loss_var

    # R_feature loss
    rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
    loss_distr = sum([mod.r_feature * rescale[idxxx] for (idxxx, mod) in enumerate(loss_r_feature_layers)])
    loss = loss + bn_reg_scale*loss_distr # best for noise before BN

    # l2 loss
    if 1:
        loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

    return loss
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.m = mean
        self.v = var
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def generate_random_task_data_fromDI(args,allData,allData_label):
    specific=random.sample(range(dataset_classnum[args.dataset]), args.way_pretrain)
    specific = [s + bias[args.dataset] for s in specific]
    support=[]
    support_label_relative=[]
    query=[]
    query_label_relative=[]
    for c_rel,c_abs in enumerate(specific):
        select_data=allData[allData_label==c_abs]
        support.append(select_data[:args.num_sup_train])
        support_label_relative.append(torch.LongTensor([c_rel]*args.num_sup_train))
        query.append(select_data[args.num_sup_train:args.num_sup_train+args.num_qur_train])
        query_label_relative.append(torch.LongTensor([c_rel]*args.num_qur_train))
    return torch.cat(support),torch.cat(support_label_relative).cuda(args.device),torch.cat(query),torch.cat(query_label_relative).cuda(args.device)
# def label_abs2relative(specific, label_abs):
#     trans = dict()
#     for relative, abs in enumerate(specific):
#         trans[abs] = relative
#     label_relative = []
#     for abs in label_abs:
#         label_relative.append(trans[abs.item()])
#     return torch.LongTensor(label_relative)

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
def get_random_task_from_data(args,data,label_relative):
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