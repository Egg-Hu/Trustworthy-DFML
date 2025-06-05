
import os
import shutil
import sys

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


class RANDOM():
    def __init__(self,args):
        super(RANDOM, self).__init__()
        self.args=args
        #file
        feature1='{}'.format(args.method)
        feature2_1 = '{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset, args.pre_backbone, args.backbone,args.APInum, args.oodP,args.fakeP, args.fakefrom)
        feature2_2 = '{}wPre_{}S_{}Q_{}teststep_{}innerlr'.format(
            args.way_pretrain,args.num_sup_train,args.num_qur_train,
            args.test_inner_update_num, args.inner_lr)
        feature2=feature2_1+'_'+feature2_2
        feature2 = feature2 + '_{}'.format(args.extra)
        self.checkpoints_path = './checkpoints_random/' + feature1 + '/' + feature2
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
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x):
        scores  = self.model(x)
        return scores
    def train_once(self,task_id=None):
        return 0
    def train_loop(self):
        test_acc_avg, test_pm = self.test_loop(self.test_loader)
        self.logger.info('[random], [TestAcc]:{} +- {}.'.format(test_acc_avg, test_pm))
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
    def test_loop(self,test_loader):
        test_acc_all = []
        for test_batch in test_loader:
            data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc=self.test_once(support=support,support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
            test_acc_all.append(test_acc)
        test_acc_avg, pm = compute_confidence_interval(test_acc_all)
        return test_acc_avg,pm
