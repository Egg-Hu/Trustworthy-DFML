from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class Scheduler(nn.Module):
    def __init__(self,args, N, grad_indexes, use_deepsets=True):
        super(Scheduler, self).__init__()
        self.percent_emb = nn.Embedding(100, 5)
        self.modelid_emb = nn.Embedding(700, 20)
        self.args=args
        self.grad_lstm = nn.LSTM(N, 10, 1, bidirectional=True)
        self.loss_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
        self.grad_lstm_2 = nn.LSTM(N, 10, 1, bidirectional=True)
        self.grad_indexes = grad_indexes
        self.use_deepsets = use_deepsets
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        input_dim = 85
        if use_deepsets:
            self.h = nn.Sequential(nn.Linear(input_dim, 20), nn.Tanh(), nn.Linear(20, 10))
            self.fc1 = nn.Linear(input_dim + 10, 20)
        else:
            self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        self.tasks = []
        self.resource=None

    def forward(self, l, input, pt,model_ids):
        x_percent = self.percent_emb(pt)
        modelid=self.modelid_emb(model_ids)

        grad_output_1, (hn, cn) = self.grad_lstm(input[0].reshape(1, len(input[0]), -1))
        grad_output_1 = grad_output_1.sum(0)
        grad_output_2, (hn, cn) = self.grad_lstm_2(input[1].reshape(1, len(input[1]), -1))
        grad_output_2 = grad_output_2.sum(0)
        grad_output = torch.cat((grad_output_1, grad_output_2), dim=1)

        loss_output, (hn, cn) = self.loss_lstm(l.reshape(1, len(l), 1))
        loss_output = loss_output.sum(0)
        x = torch.cat((x_percent,modelid, grad_output, loss_output), dim=1)

        if self.use_deepsets:
            x_C = (torch.sum(x, dim=1).unsqueeze(1) - x) / (len(x) - 1)
            x_C_mapping = self.h(x_C)
            x = torch.cat((x, x_C_mapping), dim=1)
            z = torch.tanh(self.fc1(x))
        else:
            z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        return z

    def sample_task(self, prob, size, replace=True):
        self.m = Categorical(prob)
        p = prob.detach().cpu().numpy()
        if len(np.where(p > 0)[0]) < size:
            print("exceptionally all actions")
            actions = torch.tensor(np.where(p > 0)[0])
        else:
            actions = np.random.choice(np.arange(len(prob)), p=p / np.sum(p), size=size,
                                       replace=replace)
            actions = [torch.tensor(x).cuda() for x in actions]
        return actions

    # def compute_loss(self, selected_tasks_idx, maml):
    #     task_losses = []
    #     for task_idx in selected_tasks_idx:
    #         x1, y1, x2, y2 = self.tasks[task_idx]
    #         x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
    #                          x2.squeeze(0).cuda(), y2.squeeze(0).cuda()
    #         loss_val, acc_val = maml(x1, y1, x2, y2)
    #         task_losses.append(loss_val)
    #     return torch.stack(task_losses)

    # def get_weight(self,task_losses,support_query_losses,model, pt, detach=False, return_grad=False):
    #     task_acc = []
    #     task_losses_new = []
    #     input_embedding_norm = []
    #     input_embedding_cos = []
    #     query_losses = []
    #     for candidate_id in range(len(task_losses)):
    #
    #         loss_support,loss_query=support_query_losses[candidate_id]
    #
    #         if return_grad: query_losses.append(loss_query.item())
    #
    #         fast_weights = OrderedDict(model.named_parameters())
    #         task_grad_support = torch.autograd.grad(loss_support, fast_weights.values(), create_graph=False)
    #         task_grad_query = torch.autograd.grad(loss_query, fast_weights.values(), create_graph=False)
    #         task_grad = task_grad_query + task_grad_support
    #
    #         task_layer_wise_grad_cos = []
    #         task_layer_wise_grad_norm = []
    #         for i in range(len(task_grad)):
    #             if i in self.grad_indexes:
    #                 task_layer_wise_grad_cos.append(self.cosine(task_grad_support[i].flatten().unsqueeze(0), task_grad_query[i].flatten().unsqueeze(0)))
    #                 task_layer_wise_grad_norm.append(task_grad[i].norm())
    #
    #         del fast_weights
    #         del task_grad_support
    #         del task_grad_query
    #         del task_grad
    #
    #         task_layer_wise_grad_norm = torch.stack(task_layer_wise_grad_norm)
    #         task_layer_wise_grad_cos = torch.stack(task_layer_wise_grad_cos)
    #         input_embedding_norm.append(task_layer_wise_grad_norm.detach())
    #         input_embedding_cos.append(task_layer_wise_grad_cos.detach())
    #
    #     task_losses = torch.stack(task_losses)
    #
    #     #task_layer_inputs = [torch.stack(input_embedding_norm).cuda(), torch.stack(input_embedding_cos).cuda()]
    #     task_layer_inputs = [torch.stack().cuda(), torch.stack(input_embedding_cos).cuda()]
    #
    #     weight = self.forward(task_losses, task_layer_inputs,
    #                           torch.tensor([pt]).long().repeat(len(task_losses)).cuda())
    #     if detach:
    #         weight = weight.detach()
    #
    #     if return_grad:
    #         return task_losses, task_acc, weight, task_layer_inputs, query_losses
    #     else:
    #         return task_losses, task_acc, weight
    def get_weight(self,task_losses,task1_task2_losses,train_val_losses,model, pt,model_ids, detach=False):
        task_acc = []
        input_embedding_task_cos = []
        input_embedding_set_cos = []
        for candidate_id in range(len(task_losses)):

            task1_loss,task2_loss=task1_task2_losses[candidate_id]
            train_loss,val_loss=train_val_losses[candidate_id]


            fast_weights = list(model.parameters())
            if self.args.teacherMethod == 'protonet':
                fast_weights=fast_weights[:-2]
            task1_grad = torch.autograd.grad(task1_loss, fast_weights, create_graph=False,retain_graph=False)
            task2_grad = torch.autograd.grad(task2_loss, fast_weights, create_graph=False,retain_graph=False)
            train_grad = torch.autograd.grad(train_loss, fast_weights, create_graph=False,retain_graph=False)
            val_grad = torch.autograd.grad(val_loss, fast_weights, create_graph=False,retain_graph=False)

            task_layer_wise_grad_cos = []
            set_layer_wise_grad_cos = []
            for g_id in self.grad_indexes:
                task_layer_wise_grad_cos.append(self.cosine(task1_grad[g_id].flatten().unsqueeze(0), task2_grad[g_id].flatten().unsqueeze(0)))
                set_layer_wise_grad_cos.append(self.cosine(train_grad[g_id].flatten().unsqueeze(0), val_grad[g_id].flatten().unsqueeze(0)))

            del fast_weights
            del task1_grad
            del task2_grad
            del train_grad
            del val_grad

            task_layer_wise_grad_cos = torch.stack(task_layer_wise_grad_cos)
            set_layer_wise_grad_cos=torch.stack(set_layer_wise_grad_cos)
            input_embedding_task_cos.append(task_layer_wise_grad_cos.detach())
            input_embedding_set_cos.append(set_layer_wise_grad_cos.detach())

        task_losses = torch.stack(task_losses)

        #task_layer_inputs = [torch.stack(input_embedding_norm).cuda(), torch.stack(input_embedding_cos).cuda()]
        task_layer_inputs = [torch.stack(input_embedding_task_cos).cuda(), torch.stack(input_embedding_set_cos).cuda()]
        self.resource=[task_losses, task_layer_inputs,torch.tensor([pt]).long().repeat(len(task_losses)).cuda(self.args.device),torch.tensor(model_ids).cuda(self.args.device)]
        weight = self.forward(task_losses, task_layer_inputs,torch.tensor([pt]).long().repeat(len(task_losses)).cuda(self.args.device),torch.tensor(model_ids).cuda(self.args.device))
        if detach:
            weight = weight.detach()
        return task_losses, task_acc, weight