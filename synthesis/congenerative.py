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
def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class ConGenerativeSynthesizer(BaseSynthesis):
    def __init__(self, args,teacher, student, nz, iterations=1,
                 lr_g=1e-3,
                 bn=0, oh=0,
                 transform_no_toTensor=None, device='cpu'
                 # TODO: FP16 and distributed training 
                 ):
        super(ConGenerativeSynthesizer, self).__init__(teacher, student)
        self.args=args
        self.iterations = iterations
        self.nz = nz

        self.transform_no_toTensor = transform_no_toTensor

        # scaling factors
        self.lr_g = lr_g
        self.bn = bn
        self.oh = oh

        self.device = device


    def train_once(self):
        # generator
        self.generator = self.generator.to(self.device).train()
        #reset_model(self.generator)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        self.generator.train()
        self.teacher.eval()
        # hooks for deepinversion regularization
        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))
        for it in range(self.iterations):
            y_train=(list(range(self.args.way_pretrain))) * (128)
            np.random.shuffle(y_train)
            y_train = torch.LongTensor(y_train).cuda(self.device)
            z = torch.randn( size=(len(y_train), self.nz), device=self.device)
            inputs = self.generator(z,y_train)
            inputs = self.transform_no_toTensor(inputs)
            #inputs_change = self.transform_no_toTensor(inputs)
            t_out = self.teacher(inputs)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, y_train)
            loss = self.bn * loss_bn + self.oh * loss_oh
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    @torch.no_grad()
    def synthesize(self,targets):
        self.generator.eval()
        targets=targets.cuda(self.device)
        z = torch.randn( size=(len(targets), self.nz), device=self.device )
        inputs = self.generator(z,targets)
        return inputs.detach()