import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from .base import BaseSynthesis
from .hooks import DeepInversionHook
from black_box_tool import one_hot
import numpy as np


class GenerativeSynthesizer(BaseSynthesis):
    def __init__(self, args,teacher, student, nz, iterations=1,
                 lr_g=1e-3,
                 bn=0, oh=0,balance=0,
                 transform_no_toTensor=None, device='cpu'
                 # TODO: FP16 and distributed training 
                 ):
        super(GenerativeSynthesizer, self).__init__(teacher, student)
        self.args=args
        self.iterations = iterations
        self.nz = nz

        self.transform_no_toTensor = transform_no_toTensor

        # scaling factors
        self.lr_g = lr_g
        self.bn = bn
        self.oh = oh
        self.balance=balance

        self.device = device


    def train_once(self):
        # generator
        self.generator = self.generator.to(self.device).train()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        self.generator.train()
        self.teacher.eval()
        # hooks for deepinversion regularization
        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))
        for it in range(self.iterations):
            z = torch.randn(size=(128, self.nz), device=self.device)
            inputs = self.generator(z)
            inputs=self.transform_no_toTensor(inputs)
            t_out = self.teacher(inputs)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])
            #loss_act = - t_feat.abs().mean()
            p = F.softmax(t_out, dim=1).mean(0)
            loss_balance = (p * torch.log(p)).sum()  # maximization
            loss = self.bn * loss_bn + self.oh * loss_oh + self.balance * loss_balance
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    @torch.no_grad()
    def synthesize(self,targets):
        self.generator.eval()
        z = torch.randn( size=(len(targets), self.nz), device=self.device )
        inputs = self.generator(z)
        return inputs.detach()