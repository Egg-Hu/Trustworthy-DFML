import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from .hooks import DeepInversionHook
from .criterions import jsdiv, get_image_prior_losses
from ._utils import ImagePool2, DataIter, clip_images

def jitter_and_flip(inputs_jit, lim=1./8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit

class DeepInvSyntheiszer(BaseSynthesis):
    def __init__(self,args, teacher, student, img_size,
                 iterations=1000, lr_g=0.1,
                 synthesis_batch_size=128,
                 adv=0.0, bn=1, oh=1, tv=1e-5, l2=0.0, 
                 save_dir='run/deepinversion', transform=None,
                 normalizer=None, device='cpu',num_classes=64,c_abs_list=None,max_batch_per_class=20):
        super(DeepInvSyntheiszer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"

        self.args=args
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.c_abs_list=c_abs_list
        self.normalizer = normalizer
        self.num_classes=num_classes
        self.transform=transform
        self.data_pool = ImagePool2(args=self.args,root=self.save_dir, num_classes=self.num_classes, transform=self.transform,max_batch_per_class=max_batch_per_class)
        self.synthesis_batch_size = synthesis_batch_size

        # Scaling factors
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.tv = tv
        self.l2 = l2

        self.device = device


    def synthesize(self, targets=None, student=None, mode=None, c_num=5, support=None,add=True):
        # setup hooks for BN regularization
        self.synthesis_batch_size = len(targets) // c_num
        self.hooks = []
        self.teacher.eval()
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))
        assert len(self.hooks) > 0, 'input model should contains at least one BN layer for DeepInversion'
        #kld_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        best_cost = 1e6
        best_inputs = None
        targets = torch.LongTensor(targets).to(self.device)
        inputs = torch.randn( size=[len(targets), *self.img_size], device=self.device ).requires_grad_()

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.iterations )

        best_inputs = inputs.data
        for it in range(self.iterations):
            inputs_aug = jitter_and_flip(inputs)
            t_out = self.teacher(inputs_aug)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0:
                s_out = student(inputs_aug)
                loss_adv = -jsdiv(s_out, t_out, T=3)
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss_tv = get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv * loss_tv + self.l2 * loss_l2
            
            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)
        # if self.student != None:
        #     self.student.train()
        # save best inputs and reset data loader
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
            # save best inputs and reset data iter
        if add == True:
            self.data_pool.add(imgs=best_inputs, c_abs_list=self.c_abs_list,
                           synthesis_batch_size_per_class=self.synthesis_batch_size, mode=mode)
        # print(best_cost)
        return best_inputs

    def get_random_task(self, num_w=5, num_s=5, num_q=15,mode='split'):
        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q,mode=mode)

    def get_specific_task(self, num_w=5, num_s=5, num_q=15, specific=None):
        assert specific != None, "specific can not be None!"
        return self.data_pool.get_specific_task(num_w=num_w, num_s=num_s, num_q=num_q, specific=specific)

    def sample(self, num_per_class):
        return self.data_pool.sample(num_per_class)