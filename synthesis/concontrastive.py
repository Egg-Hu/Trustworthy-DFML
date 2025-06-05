import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from torch.nn import CrossEntropyLoss
from torchvision import transforms

from black_box_tool import kldiv

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from black_box_tool import Timer
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSynthesis
from .hooks import DeepInversionHook
from ._utils import ImagePool2
from kornia import augmentation


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class ConSynthesizer(BaseSynthesis):
    def __init__(self, args,teacher, student, generator, nz, num_classes, img_size,
                 iterations=200, lr_g=0.1,
                 synthesis_batch_size=128,
                 bn=1, oh=1,adv=1,
                 save_dir='run/cmi', transform=None,transform_no_toTensor=None,
                 device='cpu', c_abs_list=None,max_batch_per_class=20):
        super(ConSynthesizer, self).__init__(teacher, student)
        self.args=args
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.bn = bn
        self.oh = oh
        self.adv=adv
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.c_abs_list=c_abs_list
        self.transform = transform
        #self.data_pool = ImagePool(root=self.save_dir,num_classes=self.num_classes,transform=self.transform)
        self.data_pool = ImagePool2(args=self.args,root=self.save_dir, num_classes=self.num_classes, transform=self.transform,max_batch_per_class=max_batch_per_class)
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []
        self.transform_no_toTensor=transform_no_toTensor
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

    def synthesize(self, targets=None, student=None, mode=None, c_num=5, support=None,iteration_id=0,add=True):#targets must be [0,1,2,3,4,0,1,2,3,4,...]
        self.synthesis_batch_size = len(targets) // c_num
        self.hooks = []
        self.teacher.eval()
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))
        ########################
        best_cost = 1e6
        best_inputs = None
        z = torch.randn(size=(len(targets), self.nz), device=self.device)
        targets = torch.LongTensor(targets).to(self.device)
        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}], self.lr_g, betas=[0.5, 0.999])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.1*self.lr)
        timeperit = Timer()
        inputs = self.generator(z,targets)
        best_inputs=inputs.data
        #smooth = random.uniform(0.0, 0.5)
        #smooth=(iteration_id*(1.0-1.0/self.args.way_pretrain))/(self.args.episode_train/self.args.episode_batch)
        # print('smooth:', smooth)
        t_out_pre=None
        for it in range(self.iterations):
            inputs = self.generator(z,targets)
            inputs_change = self.transform_no_toTensor(inputs)
            #############################################
            # Loss
            #############################################
            if self.args.ZO == False:
                t_out = self.teacher(inputs_change)
                if self.args.noBnLoss == False:
                    loss_bn = sum([h.r_feature for h in self.hooks])
                else:
                    loss_bn = 0
                # smooth = random.uniform(0.0, 0.5)
                # smooth=(iteration_id*(1.0-1.0/self.args.way_pretrain))/(self.args.episode_train/self.args.episode_batch)
                # print('smooth:', smooth)
                loss_oh = F.cross_entropy(t_out, targets)
                # if it==0:
                #     loss_oh = F.cross_entropy(t_out, targets,label_smoothing=smooth)
                #     t_out_pre=t_out.detach()
                # else:
                #     loss_oh = F.cross_entropy(t_out, t_out_pre)
                #     t_out_pre=t_out.detach()
                loss = self.bn * loss_bn + self.oh * loss_oh
                if student != None and self.adv!=0:
                    assert student != None, 'student is None!'
                    # with torch.no_grad():
                    s_out = student(inputs_change)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    # print(mask.sum())
                    # kl
                    if self.args.AdvKL:
                        loss_adv = -(kldiv(t_out, s_out.detach()).sum(1) * mask).mean()
                    # l1
                    elif self.args.AdvL1:
                        loss_adv = -(kldiv(t_out, s_out.detach(), reduction='none').sum(1) * mask).mean()
                    loss = loss + self.adv * loss_adv

            else:
                inputs_change.requires_grad_(True)
                inputs_change.retain_grad()
                # zero-order black-box
                criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda(self.device)
                with torch.no_grad():
                    m, sigma = 0, 100  # mean and standard deviation
                    mu = torch.tensor(self.args.mu).cuda(self.device)
                    q = torch.tensor(self.args.q).cuda(self.device)
                    batch_size = inputs_change.size()[0]
                    channel = inputs_change.size()[1]
                    h = inputs_change.size()[2]
                    w = inputs_change.size()[3]
                    d = channel * h * w
                    # original
                    original_t_out = self.teacher(inputs_change)
                    original_loss_oh = criterion(original_t_out, targets)
                    if student != None:
                        with torch.no_grad():
                            s_out = student(inputs_change)
                        mask = (s_out.max(1)[1] == original_t_out.max(1)[1]).float()
                        original_loss_adv = -(F.l1_loss(original_t_out, s_out.detach(), reduction='none').sum(1) * mask)
                        assert len(original_loss_adv.shape) == 1, 'error'
                    inputs_change_flatten = torch.flatten(inputs_change, start_dim=1).cuda(self.device)
                    # #serial
                    # grad_est = torch.zeros(batch_size, d).cuda(self.device)
                    # # ZO Gradient Estimation
                    # for k in range(self.args.q):
                    #     # Obtain a random direction vector
                    #     u = torch.normal(m, sigma, size=(batch_size, d))
                    #     u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size,d)  # dim -- careful
                    #     u = torch.div(u, u_norm).cuda()  # (batch_size, d)
                    #     #forward difference
                    #     inputs_change_flatten_q=inputs_change_flatten+mu*u
                    #     inputs_change_q=inputs_change_flatten_q.view(batch_size, channel, h, w)
                    #     t_out_q=self.teacher(inputs_change_q)
                    #     loss_oh_q=criterion( t_out_q, targets )
                    #     loss_diff = torch.tensor(loss_oh_q - original_loss_oh)#[bs]
                    #     assert len(loss_diff.shape)==1,'error'
                    #     if student!=None:
                    #         #s_out = student(inputs_change)
                    #         mask = (s_out.max(1)[1] == t_out_q.max(1)[1]).float()
                    #         loss_adv_q = -(F.l1_loss(t_out_q,s_out.detach(), reduction='none').sum(1) * mask)
                    #         assert len(loss_adv_q.shape)==1,'error'
                    #         loss_diff=loss_diff+self.adv*torch.tensor(loss_adv_q-original_loss_adv)
                    #     grad_est = grad_est + (d / q) * u * loss_diff.reshape(batch_size, 1).expand_as(u) / mu

                    # parallel
                    num_split = self.args.numsplit  # 5 for cifar
                    grad_est = torch.zeros((self.args.q // num_split) * batch_size, d).cuda(self.device)
                    new_original_loss_oh = original_loss_oh.repeat((self.args.q // num_split))
                    if student != None:
                        new_original_loss_adv = original_loss_adv.repeat((self.args.q // num_split))
                        assert len(new_original_loss_adv.shape) == 1, 'error'
                        new_s_out = s_out.repeat((self.args.q // num_split), 1)
                    assert len(new_original_loss_oh.shape) == 1, 'error'
                    new_targets = targets.repeat((self.args.q // num_split))
                    inputs_change_flatten = torch.flatten(inputs_change, start_dim=1).repeat((self.args.q // num_split),
                                                                                             1).cuda(self.device)

                    for k in range(num_split):
                        u = torch.normal(m, sigma, size=((self.args.q // num_split) * batch_size, d))
                        u_norm = torch.norm(u, p=2, dim=1).reshape((self.args.q // num_split) * batch_size, 1).expand(
                            (self.args.q // num_split) * batch_size, d)  # dim -- careful
                        u = torch.div(u, u_norm).cuda()  # (q*batch_size, d)
                        # forward difference
                        inputs_change_flatten_q = inputs_change_flatten + mu * u
                        inputs_change_q = inputs_change_flatten_q.view((self.args.q // num_split) * batch_size, channel,
                                                                       h, w)
                        t_out_q = self.teacher(inputs_change_q)
                        loss_oh_q = criterion(t_out_q, new_targets)
                        loss_diff = torch.tensor(loss_oh_q - new_original_loss_oh)
                        assert len(loss_diff.shape) == 1, 'error'
                        if student != None:
                            mask = (new_s_out.max(1)[1] == t_out_q.max(1)[1]).float()
                            loss_adv_q = -(F.l1_loss(t_out_q, new_s_out.detach(), reduction='none').sum(1) * mask)
                            assert len(loss_adv_q.shape) == 1
                            loss_diff = loss_diff + self.adv * torch.tensor(loss_adv_q - new_original_loss_adv)
                        assert loss_diff.shape[0] == (self.args.q // num_split) * batch_size
                        assert loss_diff.shape[0] == (self.args.q // num_split) * batch_size
                        grad_est = grad_est + (d / q) * u * loss_diff.reshape((self.args.q // num_split) * batch_size,
                                                                              1).expand_as(u) / mu
                        assert grad_est.shape[0] == (self.args.q // num_split) * batch_size
                    grad_est = grad_est.reshape((self.args.q // num_split), batch_size, d).sum(0)
                    assert grad_est.shape[0] == batch_size
                    # print('ETA:{}/{}'.format(
                    #     timeperit.measure(),
                    #     timeperit.measure((it+1) / (self.iterations)))
                    # )

                inputs_change_flatten = torch.flatten(inputs_change, start_dim=1).cuda(self.device)
                grad_est_no_grad = grad_est.detach()
                loss = torch.sum(inputs_change_flatten * grad_est_no_grad, dim=-1).mean()

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
            optimizer.zero_grad()
            # print('iteartions',it,'loss',loss.item())
            # print('iteration at', it,'q=',self.args.q)
            loss.backward()
            optimizer.step()
        # save best inputs and reset data iter
        if add==True:
            self.data_pool.add(imgs=best_inputs, c_abs_list=self.c_abs_list,
                               synthesis_batch_size_per_class=self.synthesis_batch_size, mode=mode)
        # print(best_cost)
        # print(list(self.generator.parameters())[0])
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        return best_inputs

    def get_random_task(self, num_w=5, num_s=5, num_q=15,mode='split'):
        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q,mode=mode)

    def get_specific_data(self, specific):
        assert specific != None, "specific can not be None!"
        return self.data_pool.get_specific_data(specific=specific)

    def sample(self, num_per_class):
        return self.data_pool.sample(num_per_class)

    def generate_add(self,targets,mode,add=False): #targets must be [0,1,2,3,4,0,1,2,3,4,...] when add==True
        self.synthesis_batch_size = len(targets) // self.args.way_train
        z = torch.randn(size=(len(targets), self.nz), device=self.device)
        self.generator.eval()
        inputs = self.generator(z, targets)
        if add==True:
            self.data_pool.add(imgs=inputs.data, c_abs_list=self.c_abs_list,
                               synthesis_batch_size_per_class=self.synthesis_batch_size, mode=mode)
        return inputs.data,targets
    def add_pool(self,imgs,label_specifics, c_spe_list,mode):
        self.data_pool.add(imgs=imgs, c_abs_list=c_spe_list, mode=mode,c_abs_targets=label_specifics)
