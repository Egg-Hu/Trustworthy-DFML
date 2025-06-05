
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


class DFPROTONET(nn.Module):
    def __init__(self,args):
        super(DFPROTONET, self).__init__()
        self.args=args
        #file
        feature1='{}'.format(args.method)
        feature2_1 = '{}_{}_{}_{}APINum_{}oodP_{}fakeP_{}fakefrom'.format(args.dataset, args.pre_backbone, args.backbone,args.APInum, args.oodP,args.fakeP, args.fakefrom)
        feature2_2 = '{}wPre_{}S_{}Q_{}lr_{}batch_{}InvM_{}Git_{}Glr'.format(
            args.way_pretrain, args.num_sup_train,args.num_qur_train,args.outer_lr, args.episode_batch,args.inversionMethod,args.generate_iterations,args.Glr)
        feature2=feature2_1+'_'+feature2_2
        feature2 = feature2 + '_{}'.format(args.extra)
        self.checkpoints_path = './checkpoints/' + feature1 + '/' + feature2
        self.writer_path = os.path.join(self.checkpoints_path, 'writer')
        if os.path.exists(self.writer_path):
            shutil.rmtree(self.writer_path)
        os.makedirs(self.writer_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.logger = get_logger(feature1 + '/' + feature2, output=self.checkpoints_path + '/' + 'log.txt')
        _, _, self.test_loader = get_dataloader(self.args)
        #meta model
        set_maml(False)
        self.model=get_model(args=args,type=args.backbone,set_maml_value=False,arbitrary_input=False).cuda(self.args.device)
        self.model.trunk[-1].bias.data.fill_(0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.outer_lr)
        self.loss_fn = nn.CrossEntropyLoss()

        #synthesizer
        nz = 256
        generator = Generator(nz=nz, ngf=64, img_size=args.img_size, nc=args.channel).cuda(args.device)
        self.transform_no_toTensor = get_transform_no_toTensor(args)
        self.transform = get_transform(args)
        if os.path.exists('./datapool/' + os.path.join(feature1, feature2)):
            shutil.rmtree('./datapool/' + os.path.join(feature1, feature2))
            print('remove')
        os.makedirs('./datapool/' + os.path.join(feature1, feature2), exist_ok=True)
        datapool_path = './datapool/' + os.path.join(feature1, feature2)
        max_batch_per_class = 20
        if args.inversionMethod == 'cmi':
            self.synthesizer = Synthesizer(args, None, None, generator, nz=nz, num_classes=500,
                                      img_size=(args.channel, args.img_size, args.img_size),
                                      iterations=args.generate_iterations, lr_g=args.Glr,
                                      synthesis_batch_size=30,
                                      bn=1.0, oh=1.0, adv=args.adv,
                                      save_dir=datapool_path, transform=self.transform,
                                      transform_no_toTensor=self.transform_no_toTensor,
                                      device=args.gpu, c_abs_list=None, max_batch_per_class=max_batch_per_class)
        elif args.inversionMethod == 'deepinv':
            self.synthesizer = DeepInvSyntheiszer(args, None, None, img_size=(args.channel, args.img_size, args.img_size),
                                             iterations=20000, lr_g=args.Glr,
                                             synthesis_batch_size=30,
                                             adv=0.0, bn=0.01, oh=1, tv=1e-4, l2=0.0,
                                             save_dir=datapool_path, transform=self.transform,
                                             normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]), device=args.gpu,
                                             num_classes=500, c_abs_list=None,
                                             max_batch_per_class=max_batch_per_class)
    def forward(self,x):
        scores  = self.model(x)
        return scores
    def train_once(self,task_id=None):
        #generation
        if task_id%self.args.generate_interval==1:
            #generate support
            teacher, specific, acc,teacher_dataset=get_random_teacher(args=self.args)
            teacher.eval()
            self.synthesizer.teacher = teacher
            self.synthesizer.c_abs_list = specific
            support_tensor = self.synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * self.args.num_sup_kd), student=None, mode='support',c_num=len(specific))
            save_image_batch(support_tensor, self.checkpoints_path + '/support.png', col=self.args.way_pretrain)
            #generate query
            query_tensor = self.synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * (self.args.num_qur_kd)), student=None,mode='query', c_num=len(specific))
        support_data, support_label_abs, query_data, query_label_abs, specific = self.synthesizer.get_random_task(num_w=self.args.way_train, num_s=self.args.num_sup_train, num_q=self.args.num_qur_train)
        support_label = label_abs2relative(specific, support_label_abs).cuda(self.args.device)
        query_label = label_abs2relative(specific, query_label_abs).cuda(self.args.device)
        support, support_label, query, query_label = support_data.cuda(self.args.device), support_label.cuda(self.args.device), query_data.cuda(self.args.device), query_label.cuda(self.args.device)
        #proto
        z_support = self.model.getFeature(support)
        z_query = self.model.getFeature(query)
        z_support = z_support.contiguous().view(self.args.way_train * self.args.num_sup_train, -1)
        z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.args.way_train, self.args.num_sup_train, -1).mean(1)
        z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return self.loss_fn(scores,query_label)
    def train_loop(self):
        timer = Timer()
        with SummaryWriter(self.writer_path) as writer:
            loss_batch = []
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
                    self.test_loop(self.test_loader,task_id)
                    print('ETA:{}/{}'.format(
                        timer.measure(),
                        timer.measure((task_id) / (self.args.episode_train)))
                    )
    def test_once(self,support,support_label_relative,query, query_label_relative):
        # proto
        z_support = self.model.getFeature(support)
        z_query = self.model.getFeature(query)

        z_support = z_support.contiguous().view(self.args.way_train * self.args.num_sup_train, -1)
        z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)

        z_support = z_support.contiguous()
        #1234512345
        z_proto = z_support.view(self.args.way_train, self.args.num_sup_train, -1).t().mean(1)
        z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        correct=0
        total=0
        prediction = torch.max(scores, 1)[1]
        correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
        total = total + len(query_label_relative)
        acc=1.0*correct/total*100.0
        return acc
    def test_loop(self,test_loader,task_id):
        test_acc_all = []
        test_acc_max=0
        max_it=0
        max_pm=0
        for test_batch in test_loader:
            data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc=self.test_once(support=support,support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
            test_acc_all.append(test_acc)
        test_acc_avg, pm = compute_confidence_interval(test_acc_all)
        if test_acc_avg > test_acc_max:
            test_acc_max = test_acc_avg
            max_it = (task_id) // self.args.episode_batch
            max_pm = pm
            torch.save(self.model.state_dict(), self.checkpoints_path + '/bestmodel_{}it.pth'.format(max_it))
        self.logger.info('[Epoch]:{}, [TestAcc]:{} +- {}. [BestEpoch]:{}, [BestTestAcc]:{} +- {}.'.format((task_id) //self.args.episode_batch, test_acc_avg, pm, max_it, test_acc_max, max_pm))
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