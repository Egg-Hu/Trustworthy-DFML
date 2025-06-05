from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from .hooks import DeepInversionHook, InstanceMeanHook
from .criterions import jsdiv, get_image_prior_losses, kldiv
from ._utils import ImagePool, DataIter, clip_images
import collections
from torchvision import transforms
from kornia import augmentation

from ..tool import label_abs2relative
from ..methods.maml import Maml


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class ADVSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, 
                 feature_layers=None ,
                 iterations=200, lr_g=0.1,
                 synthesis_batch_size=128,
                 adv=0.0, bn=1, oh=1, cr=0.8, cr_T=0.1,
                 save_dir='run/cmi', transform=None,normalizer=None,
                 device='cpu', c_abs_list=None):
        super(ADVSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.c_abs_list=c_abs_list
        self.normalizer=normalizer
        self.transform = transform
        self.data_pool = ImagePool(root=self.save_dir,num_classes=self.num_classes,transform=self.transform)
        
        self.generator = generator.to(device).train()
        self.device = device
        self.aug=transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )


    def synthesize(self, targets=None,adv=0.0,args=None,model_maml=None):
        maml=Maml(args)
        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))
        ########################
        if self.student!=None:
            self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        
        #inputs = torch.randn( size=(self.synthesis_batch_size, *self.img_size), device=self.device ).requires_grad_()
        best_inputs = None
        z = torch.randn(size=(len(targets), self.nz), device=self.device).requires_grad_()
        targets = targets.to(self.device)

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.1*self.lr)
        for it in range(self.iterations):
            inputs = self.generator(z)

            #############################################
            # Inversion Loss
            #############################################
            global_view=self.aug(inputs)
            t_out = self.teacher(global_view)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            loss_inv = self.bn * loss_bn + self.oh * loss_oh
            if adv!=0:
                support, support_label_abs, query, query_label_abs, specific=self.get_random_task_from_inputs(num_w=args.way_train,num_s=args.num_sup_train,num_q=args.num_qur_train)
                support = support.cuda()
                query = query.cuda()
                support_label_relative = label_abs2relative(specific, support_label_abs).cuda()
                query_label_relative = label_abs2relative(specific, query_label_abs).cuda()
                loss_outer, _ = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                           query=query, query_label=query_label_relative, criteria=nn.CrossEntropyLoss(),
                                           device=torch.device('cuda:{}'.format(args.gpu)), mode='train')

            loss=loss_inv-adv*loss_outer


            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
                    best_it=it
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.student != None:
            self.student.train()
        # save best inputs and reset data iter
        self.data_pool.add( imgs=best_inputs ,c_abs_list=self.c_abs_list,synthesis_batch_size_per_class=self.synthesis_batch_size)
        #print('best it:',it)
        return {"synthetic": best_inputs}

    def get_random_task(self, num_w=5, num_s=5, num_q=15):
        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q)

    def get_specific_task(self, num_w=5, num_s=5, num_q=15, specific=None):
        if specific is None:
            specific = [0, 1, 2, 3, 4]
        return self.data_pool.get_specific_task(num_w=num_w, num_s=num_s, num_q=num_q, specific=specific)

    def sample(self, num_per_class):
        return self.data_pool.sample(num_per_class)
    def get_random_task_from_inputs(self, num_w=5, num_s=5, num_q=15,inputs=None,targets=None):

        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q)