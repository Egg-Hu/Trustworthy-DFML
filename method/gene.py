
import os
import shutil
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import torch.nn as nn

from black_box_tool import get_transform_no_toTensor, get_transform, Normalizer, NORMALIZE_DICT, \
    label_abs2relative
from generator import Generator
from synthesis import Synthesizer, DeepInvSyntheiszer
from tensorboardX import SummaryWriter
from black_box_tool import set_maml, get_model, get_random_teacher, kldiv, data2supportquery, \
    compute_confidence_interval, get_dataloader, Timer
from logger import get_logger
from synthesis._utils import save_image_batch

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class GENE(nn.Module):#each class correspond to a generator, self-supervised task
    def __init__(self,args):
        super(GENE, self).__init__()
        self.args=args
        #file
        feature1='{}'.format(args.method)
        feature2_1 = '{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset, args.pre_backbone, args.backbone,args.APInum, args.oodP,args.fakeP, args.fakefrom)
        feature2_2 = '{}wPre_{}S_{}Q_{}steptrain_{}teststep_{}innerlr_{}outerlr_{}batch_{}InvM_{}Git_{}Glr'.format(
            args.way_pretrain,args.num_sup_train,args.num_qur_train, args.inner_update_num,
            args.test_inner_update_num, args.inner_lr, args.outer_lr, args.episode_batch,args.inversionMethod,args.generate_iterations,args.Glr)
        feature2=feature2_1+'_'+feature2_2
        feature2 = feature2 + '_{}'.format(args.extra)
        if args.approx:
            feature2 = feature2 + '_1Order'
        self.checkpoints_path = './checkpoints/' + feature1 + '/' + feature2
        self.writer_path = os.path.join(self.checkpoints_path, 'writer')
        if os.path.exists(self.writer_path):
            shutil.rmtree(self.writer_path)
        os.makedirs(self.writer_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.logger = get_logger(feature1 + '/' + feature2, output=self.checkpoints_path + '/' + 'log.txt')
        _, _, self.test_loader = get_dataloader(self.args)
        #meta model
        set_maml(True)
        self.model=get_model(args=args,type=args.backbone,set_maml_value=True,arbitrary_input=False).cuda(self.args.device)
        set_maml(False)
        self.model.trunk[-1].bias.data.fill_(0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.outer_lr)
        self.loss_fn = nn.CrossEntropyLoss()

        #synthesizer
        nz = 256
        self.generator = Generator(nz=nz, ngf=64, img_size=args.img_size, nc=args.channel).cuda(args.device)
        self.transform_no_toTensor = get_transform_no_toTensor(args)
        self.transform = get_transform(args)

    def forward(self,x):
        scores  = self.model(x)
        return scores
    def train_once(self,task_id=None):
        support_x = []
        query_x = []
        support_y_rel = []
        query_y_rel = []
        for c in range(self.args.way_train):
            reset_model(self.generator)
            z = torch.randn(size=(self.args.num_sup_train+self.args.num_qur_train, 256), device=self.args.device)
            select_image=self.transform_no_toTensor((self.generator(z)).data)
            support_x.append(select_image[:self.args.num_sup_train])
            query_x.append(select_image[self.args.num_sup_train:(self.args.num_sup_train + self.args.num_qur_train)])
            support_y_rel.append(torch.LongTensor([c] * self.args.num_sup_train))
            query_y_rel.append(torch.LongTensor([c] * self.args.num_sup_train))
        support=torch.cat(support_x,dim=0).cuda(self.args.device)
        support_label=torch.cat(support_y_rel,dim=0).cuda(self.args.device)
        query = torch.cat(support_x, dim=0).cuda(self.args.device)
        query_label = torch.cat(query_y_rel, dim=0).cuda(self.args.device)
        result = list(zip(support, support_label))  # 打乱的索引序列
        np.random.shuffle(result)
        support, support_label = zip(*result)
        support=torch.stack(list(support),dim=0).cuda(self.args.device)
        support_label=torch.tensor(list(support_label)).cuda(self.args.device)
        result = list(zip(query, query_label))  # 打乱的索引序列
        np.random.shuffle(result)
        query, query_label = zip(*result)
        query = torch.stack(list(query), dim=0).cuda(self.args.device)
        query_label = torch.tensor(list(query_label)).cuda(self.args.device)
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
        return loss_outer
    def train_loop(self):
        timer = Timer()
        with SummaryWriter(self.writer_path) as writer:
            loss_batch = []
            test_acc_max = 0
            max_it = 0
            max_pm = 0
            for task_id in range(self.args.start_id, self.args.episode_train + 1):
                loss=self.train_once(task_id)
                loss_batch.append(loss)
                if task_id % self.args.episode_batch == 0:
                    loss = torch.stack(loss_batch).sum(0)
                    loss_batch = []
                    writer.add_scalar(tag='train_loss', scalar_value=loss.item(),global_step=(task_id) // self.args.episode_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # val
                if task_id % self.args.val_interval == 0:
                    test_acc_avg,test_pm=self.test_loop(self.test_loader,task_id)
                    if test_acc_avg > test_acc_max:
                        test_acc_max = test_acc_avg
                        max_it = (task_id) // self.args.episode_batch
                        max_pm = test_pm
                        torch.save(self.model.state_dict(),
                                   self.checkpoints_path + '/bestmodel_{}it.pth'.format(max_it))
                    self.logger.info('[Epoch]:{}, [TestAcc]:{} +- {}. [BestEpoch]:{}, [BestTestAcc]:{} +- {}.'.format(
                        (task_id) // self.args.episode_batch, test_acc_avg, test_pm, max_it, test_acc_max, max_pm))

                    print('ETA:{}/{}'.format(
                        timer.measure(),
                        timer.measure((task_id) / (self.args.episode_train)))
                    )
    def test_once(self,support,support_label_relative,query, query_label_relative):
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
        return acc
    def test_loop(self,test_loader,task_id):
        test_acc_all = []
        for test_batch in test_loader:
            data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc=self.test_once(support=support,support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
            test_acc_all.append(test_acc)
        test_acc_avg, pm = compute_confidence_interval(test_acc_all)
        return test_acc_avg,pm
