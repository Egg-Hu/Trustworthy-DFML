import shutil

import torch
from torch.utils.data import ConcatDataset, Dataset
import numpy as np 
from PIL import Image
import os, random, math
from copy import deepcopy
from contextlib import contextmanager
import sys


sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from black_box_tool import bias,OOD_list, bias_end



def get_pseudo_label(n_or_label, num_classes, device, onehot=False):
    if isinstance(n_or_label, int):
        label = torch.randint(0, num_classes, size=(n_or_label,), device=device)
    else:
        label = n_or_label.to(device)
    if onehot:
        label = torch.zeros(len(label), num_classes, device=device).scatter_(1, label.unsqueeze(1), 1.)
    return label

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(dim_feat, c, max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size
        assert k <= self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def clip_images(image_tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor
    
def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        #output_filename = output.strip('.png')
        output_filename = output[:-4]
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0).squeeze() )
            img.save(output_filename+'-%d.png'%(idx))

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

def load_yaml(filepath):
    yaml=YAML()  
    with open(filepath, 'r') as f:
        return yaml.load(f)

def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith( pos ):
                    images.append( os.path.join( dirpath, f ) )
    return images

class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root) #[ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open( self.images[idx] )
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)

class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.categories = [ int(f) for f in os.listdir( root ) ]
        images = []
        targets = []
        for c in self.categories:
            category_dir = os.path.join( self.root, str(c))
            _images = [ os.path.join( category_dir, f ) for f in os.listdir(category_dir) ]
            images.extend(_images)
            targets.extend([c for _ in range(len(_images))])
        self.images = images
        self.targets = targets
        self.transform = transform
    def __getitem__(self, idx):
        img, target = Image.open( self.images[idx] ), self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.images)

class ImagePool(object):
    def __init__(self, root,num_classes,transform):
        #self.root = os.path.abspath(root)
        self.root=root
        for c_abs in range(num_classes):
            os.makedirs(os.path.join(self.root,str(c_abs)), exist_ok=True)
        self._idx = [0]*num_classes
        self.max_batch_per_class=20
        self.ready_class=[]
        self.transform=transform

    def add(self, imgs, c_abs_list=None,synthesis_batch_size_per_class=None):
        c_abs_targets=torch.LongTensor(c_abs_list*synthesis_batch_size_per_class)
        for c_abs in c_abs_list:
            root=os.path.join(self.root,str(c_abs))
            imgs_c=imgs[c_abs_targets==c_abs]
            save_image_batch(imgs_c, os.path.join( root, "%d.png"%(self._idx[c_abs]) ), pack=False)
            self._idx[c_abs]+=1
            self._idx[c_abs]=self._idx[c_abs]%self.max_batch_per_class
            if c_abs not in self.ready_class:
                self.ready_class.append(c_abs)
        os.makedirs(os.path.join(self.root,'view'),exist_ok=True)
        #save_image_batch(imgs,os.path.join(self.root,'view/view.png'),col=5,pack=True)

    def get_random_task(self,num_w,num_s,num_q):
        select_way=random.sample(self.ready_class,num_w)
        support_x=[]
        query_x=[]
        support_y_abs=[]
        query_y_abs = []
        for c_relative,c_abs in enumerate(select_way):
            c_abs_path=os.path.join(self.root,str(c_abs))
            image_name_list = os.listdir(c_abs_path)
            image_name_list=random.sample(image_name_list,(num_s+num_q))
            #print(image_name_list)
            select_image=[self.transform(Image.open(os.path.join(c_abs_path,image_name)).convert('RGB')) for image_name in image_name_list]
            select_image=torch.stack(select_image,dim=0)
            support_x.append(select_image[:num_s])
            query_x.append(select_image[num_s:(num_s+num_q)])
            support_y_abs.append(torch.LongTensor([c_abs]*num_s))
            query_y_abs.append(torch.LongTensor([c_abs] * num_q))
        return torch.cat(support_x,dim=0),torch.cat(support_y_abs,dim=0),torch.cat(query_x,dim=0),torch.cat(query_y_abs,dim=0),select_way
    def get_specific_task(self,num_w,num_s,num_q,specific):
        select_way=specific
        support_x=[]
        query_x=[]
        support_y_abs=[]
        query_y_abs = []
        for c_relative,c_abs in enumerate(select_way):
            c_abs_path=os.path.join(self.root,str(c_abs))
            image_name_list = os.listdir(c_abs_path)
            image_name_list=random.sample(image_name_list,(num_s+num_q))
            select_image=[self.transform(Image.open(os.path.join(c_abs_path,image_name)).convert('RGB')) for image_name in image_name_list]
            select_image=torch.stack(select_image,dim=0)
            support_x.append(select_image[:num_s])
            query_x.append(select_image[num_s:(num_s+num_q)])
            support_y_abs.append(torch.LongTensor([c_abs]*num_s))
            query_y_abs.append(torch.LongTensor([c_abs] * num_q))
        return torch.cat(support_x,dim=0),torch.cat(support_y_abs,dim=0),torch.cat(query_x,dim=0),torch.cat(query_y_abs,dim=0)
    def sample(self,num_perclass):
        data=[]
        label=[]
        for c_abs in self.ready_class:
            c_abs_path=os.path.join(self.root,str(c_abs))
            image_name_list = os.listdir(c_abs_path)
            image_name_list=random.sample(image_name_list,num_perclass)
            select_image=[self.transform(Image.open(os.path.join(c_abs_path,image_name)).convert('RGB')) for image_name in image_name_list]
            select_image = torch.stack(select_image, dim=0)
            data.append(select_image)
            label.append(torch.LongTensor([c_abs]*num_perclass))
        return torch.cat(data, dim=0), torch.cat(label, dim=0)

class ImagePool2(object):
    def __init__(self,args, root,num_classes,transform,max_batch_per_class):
        #self.root = os.path.abspath(root)
        self.args=args
        self.root_all=os.path.join(root,'all')
        self.root_support=os.path.join(root,'support')
        self.root_query = os.path.join(root, 'query')
        #support pool
        for c_abs in range(num_classes):
            os.makedirs(os.path.join(self.root_all, str(c_abs)), exist_ok=True)
            os.makedirs(os.path.join(self.root_support,str(c_abs)), exist_ok=True)
            os.makedirs(os.path.join(self.root_query, str(c_abs)), exist_ok=True)
        self._idx = [0]*num_classes
        self.max_batch_per_class=max_batch_per_class
        self.ready_class={name:[] for name,start in bias.items()}
        self.transform=transform

    def add(self, imgs, c_abs_list=None,synthesis_batch_size_per_class=None,mode=None,c_abs_targets=None):
        if c_abs_targets==None:
            c_abs_targets=torch.LongTensor(c_abs_list*synthesis_batch_size_per_class)
        else:
            pass
        for c_abs in c_abs_list:
            if mode=='support':
                root=os.path.join(self.root_support,str(c_abs))
            if mode=='query':
                root=os.path.join(self.root_query,str(c_abs))
            if mode=='all':
                root = os.path.join(self.root_all, str(c_abs))
            # print(c_abs_targets.shape)
            # print((c_abs_targets==c_abs).shape)
            imgs_c=imgs[c_abs_targets==c_abs]
            #print('data_pool save at:',root)
            save_image_batch(imgs_c, os.path.join( root, "%d.png"%(self._idx[c_abs]) ), pack=False)
            self._idx[c_abs]+=1
            self._idx[c_abs]=self._idx[c_abs]%self.max_batch_per_class
            for(dataset_name,end_id) in bias_end.items():
                if c_abs<=end_id:
                    if c_abs not in self.ready_class[dataset_name]:
                        self.ready_class[dataset_name].append(c_abs)
                    break
            # if c_abs not in self.ready_class:
            #     self.ready_class.append(c_abs)
        #os.makedirs(os.path.join(self.root,'view'),exist_ok=True)
        #save_image_batch(imgs,os.path.join(self.root,'view/view.png'),col=5,pack=True)
    def clear_specific(self,specific):
        c_abs_path = os.path.join(self.root_all, str(specific))
        shutil.rmtree(c_abs_path)
        assert os.path.exists(c_abs_path)==False, 'clear fail!!!'
        os.makedirs(c_abs_path,exist_ok=True)
        for (dataset_name, end_id) in bias_end.items():
            if specific <= end_id:
                if specific in self.ready_class[dataset_name]:
                    self.ready_class[dataset_name].remove(specific)
                break

    def get_random_task(self,num_w,num_s,num_q,mode='split'):
        # temp = random.random()
        # if temp <= self.args.oodP:
        #     if self.args.fakefrom == -1:
        #         #ood_dataset = random.choice(OOD_list)
        #         ready_class=[]
        #         for ood_dataset in OOD_list:
        #             ready_class=ready_class+self.ready_class[ood_dataset]
        #     else:
        #         ood_dataset = OOD_list[self.args.fakefrom]
        #         ready_class=self.ready_class[ood_dataset]
        #     #if len(self.ready_class[ood_dataset])!=0:
        #     if len(ready_class) != 0:
        #         select_way = random.sample(ready_class, num_w)
        #     else:
        #         select_way = random.sample(self.ready_class[self.args.dataset], num_w)
        # elif temp <= self.args.oodP + self.args.fakeP:
        #     select_way = random.sample(self.ready_class[self.args.dataset], num_w)
        # else:
        #     select_way = random.sample(self.ready_class[self.args.dataset], num_w)
        select_way = random.sample(self.ready_class[self.args.dataset], num_w)
        support_x=[]
        query_x=[]
        support_y_abs=[]
        query_y_abs = []
        if mode=='split':
            for c_relative,c_abs in enumerate(select_way):
                #part from S
                c_abs_path = os.path.join(self.root_support, str(c_abs))
                #print(c_abs_path)
                image_name_list = os.listdir(c_abs_path)
                image_name_list = random.sample(image_name_list, (num_s))
                if self.args.dataset!='mix':
                    if self.args.dataset == 'omniglot':
                        select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('L')) for image_name in image_name_list]
                    else:
                        select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('RGB')) for image_name in image_name_list]
                elif self.args.dataset=='mix':
                    raise NotImplementedError
                select_image = torch.stack(select_image, dim=0)
                support_x.append(select_image)
                support_y_abs.append(torch.LongTensor([c_abs] * num_s))
                #part from Q
                c_abs_path = os.path.join(self.root_query, str(c_abs))
                image_name_list = os.listdir(c_abs_path)
                image_name_list = random.sample(image_name_list, (num_q))
                if self.args.dataset != 'mix':
                    if self.args.dataset == 'omniglot':
                        select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('L')) for
                                        image_name in image_name_list]
                    else:
                        select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('RGB')) for
                                        image_name in image_name_list]
                elif self.args.dataset == 'mix':
                    raise NotImplementedError
                select_image = torch.stack(select_image, dim=0)
                query_x.append(select_image)
                query_y_abs.append(torch.LongTensor([c_abs] * num_q))
        elif mode=='all':
            for c_relative,c_abs in enumerate(select_way):
                # only from all
                c_abs_path = os.path.join(self.root_all, str(c_abs))
                image_name_list = os.listdir(c_abs_path)
                image_name_list = random.sample(image_name_list, (num_s+num_q))
                if self.args.dataset!='mix':
                    if self.args.dataset == 'omniglot':
                        select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('L')) for image_name in image_name_list]
                    else:
                        select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('RGB')) for image_name in image_name_list]
                elif self.args.dataset=='mix':
                    pass

                select_image = torch.stack(select_image, dim=0)
                support_x.append(select_image[:num_s])
                query_x.append(select_image[num_s:(num_s + num_q)])
                support_y_abs.append(torch.LongTensor([c_abs] * num_s))
                query_y_abs.append(torch.LongTensor([c_abs] * num_q))
        return torch.cat(support_x,dim=0),torch.cat(support_y_abs,dim=0),torch.cat(query_x,dim=0),torch.cat(query_y_abs,dim=0),select_way
    def get_specific_data(self,specific):
        specific_data=[]
        for c_specific in specific:
            c_abs_path = os.path.join(self.root_all, str(c_specific))
            image_name_list = os.listdir(c_abs_path)
            if len(image_name_list)!=0:
                select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('RGB')) for image_name in image_name_list]
                select_image = torch.stack(select_image, dim=0)
            else:
                select_image=None
            specific_data.append(select_image)
        return specific_data





class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass