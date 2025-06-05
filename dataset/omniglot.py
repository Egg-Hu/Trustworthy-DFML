import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import random
import torch
import os.path as osp
from PIL import Image
import torch.nn.functional as F


from torch.utils.data import Dataset
from torchvision import transforms
#from tqdm import tqdm
import numpy as np



from network import Conv4
# SPLIT_PATH = osp.join('/home/hzx/fcil/dfmeta/DFL2Ldata/omniglot/split')
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(script_dir)
SPLIT_PATH = os.path.join(parent_dir, 'DFL2Ldata/omniglot/split')

def identity(x):
    return x

class Omniglot(Dataset):
    """ Usage:
    """
    def __init__(self, setname, augment=False,noTransform=False):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        #print('csv_path:',csv_path)
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))

        self.img_size = 32
        if augment and setname == 'meta_train':
            transforms_list = [
                lambda x: x.resize((self.img_size, self.img_size), resample=Image.LANCZOS),
                # lambda x: np.reshape(x, (self.img_size, self.img_size, 3)),
                # lambda x:  np.array(x, dtype=np.float32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
                ]
        else:
            transforms_list = [
                lambda x: x.resize((self.img_size, self.img_size), resample=Image.LANCZOS),
                # lambda x: np.reshape(x, (self.img_size, self.img_size, 3)),
                # lambda x:  np.array(x, dtype=np.float32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
                ]
        if noTransform==True:
            self.transform=lambda x:np.asarray(x)
        else:
            self.transform = transforms.Compose(
                transforms_list
            )

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []#[path0,path2,path2,...]
        label = []#[0,0,0,1,2,...]
        lb = -1

        self.wnids = []

        # for l in tqdm(lines, ncols=64):
        for l in lines:
            name, wnid = l.split(',')
            path = name
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))

        return image, label
class Omniglot_Specific(Dataset):
    """ Usage:
    """

    def __init__(self, setname, specific=None, augment=False, mode='all',noTransform=False):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))
        if mode == 'all':
            data = [z[0] for z in zip(self.data, self.label) if z[1] in specific]
            label = [z[1] for z in zip(self.data, self.label) if z[1] in specific]
        elif mode == 'train':
            data=[]
            label=[]
            for select_class in specific:
                data_specific=[]
                label_specific=[]
                for z in zip(self.data, self.label):
                    if z[1] ==select_class:
                        data_specific.append(z[0])
                        label_specific.append(z[1])
                data_specific=data_specific[:int(len(data_specific)*0.8)]
                label_specific=label_specific[:int(len(label_specific)*0.8)]
                data.append(data_specific)
                label.append(label_specific)
            data=[j for i in data for j in i ]
            label=[j for i in label for j in i ]

            self.data=data
            self.label=label
        elif mode == 'test':
            data = []
            label = []
            for select_class in specific:
                data_specific = []
                label_specific = []
                for z in zip(self.data, self.label):
                    if z[1] == select_class:
                        data_specific.append(z[0])
                        label_specific.append(z[1])
                data_specific = data_specific[int(len(data_specific) * 0.8):]
                label_specific = label_specific[int(len(label_specific) * 0.8):]
                data.append(data_specific)
                label.append(label_specific)
            data = [j for i in data for j in i]
            label = [j for i in label for j in i]
            self.data = data
            self.label = label
        self.data = data
        self.label = label
        self.num_class = len(set(self.label))

        self.img_size = 32
        if augment and setname == 'meta_train':
            transforms_list = [
                lambda x: x.resize((self.img_size, self.img_size),resample=Image.LANCZOS),
                #lambda x: np.reshape(x, (self.img_size, self.img_size, 3)),
                #transforms.Resize((self.img_size, self.img_size)),
                #lambda x: np.array(x, dtype=np.float32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
                ]
        else:
            transforms_list = [
                lambda x: x.resize((self.img_size, self.img_size),resample=Image.LANCZOS),
                #lambda x: np.reshape(x, (self.img_size, self.img_size, 3)),
                #lambda x:  np.array(x, dtype=np.float32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
                ]
        if noTransform==True:
            self.transform=lambda x:np.asarray(x)
        else:
            self.transform = transforms.Compose(
                transforms_list
            )

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []#[path0,path2,path2,...]
        label = []#[0,0,0,1,2,...]
        lb = -1

        self.wnids = []

        # for l in tqdm(lines, ncols=64):
        for l in lines:
            name, wnid = l.split(',')
            path = name
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))

        return image, label
if __name__=='__main__':
    def label_abs2relative(specific, label_abs):
        trans = dict()
        for relative, abs in enumerate(specific):
            trans[abs] = relative
        label_relative = []
        for abs in label_abs:
            label_relative.append(trans[abs.item()])
        return torch.LongTensor(label_relative)


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(2022)
    classes = [3, 5, 10, 8, 33]
    train_dataset=Omniglot_Specific(setname='meta_train',specific=classes,mode='train')
    print(len(train_dataset))
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=75,shuffle=True,num_workers=8,pin_memory=True)
    test_dataset = Omniglot_Specific(setname='meta_train', specific=classes, mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=25, shuffle=True, num_workers=8,pin_memory=True)
    model=Conv4(flatten=True, out_dim=5, img_size=28,arbitrary_input=False,channel=1).cuda()
    criteria = torch.nn.CrossEntropyLoss()
    num_epoch = 60
    learning_rate = 0.001

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    for epoch in range(num_epoch):
        model.train()
        for data,label in train_loader:
            data=data.cuda()
            label=label_abs2relative(specific=classes,label_abs=label).cuda()
            logits=model(data)
            loss=criteria(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step()
        model.eval()
        correct,total=0,0
        for data,label in test_loader:
            data=data.cuda()
            label=label_abs2relative(specific=classes,label_abs=label).cuda()
            logits = model(data)
            prediction=torch.max(logits,1)[1]
            correct=correct+(prediction.cpu()==label.cpu()).sum()
            total=total+len(label)
        print('epoch:',epoch,'acc:',100*correct/total)
    #torch.save(model.state_dict(),'./model_omniglot.pth')


    # model.load_state_dict(torch.load('./model_omniglot.pth'))
    # student = Conv4(flatten=True, out_dim=5, img_size=28, arbitrary_input=False, channel=1).cuda()
    # optimizer_s = torch.optim.Adam(params=student.parameters(), lr=learning_rate)
    # for batch_step in range(600):
    #     noise=np.random.binomial(1, 0.9, (128,1,28,28))
    #     noise=torch.Tensor(noise).cuda()
    #     model.eval()
    #     student.train()
    #     with torch.no_grad():
    #         t_out=model(noise)
    #     s_out=student(noise)
    #     kl=F.kl_div(t_out.softmax(dim=-1).log(),s_out.softmax(dim=-1),reduction='sum')
    #     print(kl)
    #     optimizer_s.zero_grad()
    #     kl.backward()
    #     optimizer_s.step()
    #     student.eval()
    #     correct, total = 0, 0
    #     for data, label in test_loader:
    #         data = data.cuda()
    #         label = label_abs2relative(specific=classes, label_abs=label).cuda()
    #         logits = student(data)
    #         prediction = torch.max(logits, 1)[1]
    #         correct = correct + (prediction.cpu() == label.cpu()).sum()
    #         total = total + len(label)
    #     print('batch:', batch_step, 'student_acc:', 100 * correct / total)