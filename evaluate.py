import torch

from dfmlmd.black_box_tool import data2supportquery, compute_confidence_interval

class Evaluate():
    def __init__(self):
        super(Evaluate, self).__init__()
    def evaluate_traditional_meta(self):

    def test_once(self,support, support_label_relative, query, query_label_relative):
        self.model.zero_grad()
        fast_parameters = list(
            self.model.parameters())  # the first gradient calcuated in line 45 is based on original weight
        for weight in self.model.parameters():
            weight.fast = None

        for task_step in range(self.args.test_inner_update_num):
            scores = self.forward(support)
            loss_inner = self.loss_fn(scores, support_label_relative)
            grad = torch.autograd.grad(loss_inner, fast_parameters,
                                       create_graph=True)  # build full graph support gradient of gradient
            if self.args.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.model.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.args.inner_lr * grad[k]  # create weight.fast
                else:
                    weight.fast = weight.fast - self.args.inner_lr * grad[
                        k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(
                    weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
        # outer
        correct = 0
        total = 0
        scores = self.forward(query)
        prediction = torch.max(scores, 1)[1]
        correct = correct + (prediction.cpu() == query_label_relative.cpu()).sum()
        total = total + len(query_label_relative)
        acc = 1.0 * correct / total * 100.0

        self.model.zero_grad()
        for weight in self.model.parameters():
            weight.fast = None
        return acc


    def test_loop(self,test_loader, task_id):
        test_acc_all = []
        test_acc_max = 0
        max_it = 0
        max_pm = 0
        for test_batch in test_loader:
            data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc = self.test_once(support=support, support_label_relative=support_label_relative, query=query,
                                      query_label_relative=query_label_relative)
            test_acc_all.append(test_acc)
        test_acc_avg, pm = compute_confidence_interval(test_acc_all)
        if test_acc_avg > test_acc_max:
            test_acc_max = test_acc_avg
            max_it = (task_id) // self.args.episode_batch
            max_pm = pm
            torch.save(self.model.state_dict(), self.checkpoints_path + '/bestmodel_{}it.pth'.format(max_it))
        self.logger.info('[Epoch]:{}, [TestAcc]:{} +- {}. [BestEpoch]:{}, [BestTestAcc]:{} +- {}.'.format(
            (task_id) // self.args.episode_batch, test_acc_avg, pm, max_it, test_acc_max, max_pm))
