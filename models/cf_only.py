import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

__all__ = ['cf_only']


class Model(nn.Module):
    def __init__(self, pret=True, args=None):
        super(Model, self).__init__()
        self.dim = 2048
        option = args.arch
        nc = args.nc
        mid = 10000
        prob = 0.5

        '''feature generator'''
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50' or option == 'cf_only':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.gen = nn.Sequential(*mod)

        '''classifier'''
        layers = []
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(self.dim, mid))
        layers.append(nn.BatchNorm1d(mid, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(args.layers-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(mid, mid))
            layers.append(nn.BatchNorm1d(mid, affine=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(mid, args.nc))
        self.cls1 = nn.Sequential(*layers)
        self.cls2 = nn.Sequential(*layers)

        '''params ini'''
        for name, param in self.named_modules():
            if 'cls' in name:
                if isinstance(param, nn.Linear):
                    param.weight.data.normal_(0.0, 0.01)
                    param.bias.data.normal_(0.0, 0.01)
                if isinstance(param, nn.BatchNorm1d):
                    param.weight.data.normal_(1.0, 0.01)
                    param.bias.data.fill_(0)

    def forward(self, x):
        # generate feature
        x = self.gen(x)
        g = x.view(x.size(0), self.dim)

        # classify
        f1 = self.cls1(g)
        f2 = self.cls2(g)

        return f1, f2


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.eta = 1.0
        self.batch_size = args.batch_size
        self.measure = args.measure
        self.nc = args.nc
        self.epochs = args.epochs

    def measure_dis(self, measure, output_t1, output_t2):
        if measure == 'L1':
            return 1 * torch.mean(torch.abs(output_t1 - output_t2))

    def st_loss(self, output, area):
        batch_size = output.size(0)
        prob = F.softmax(output, dim=1)
        if area == 'left':
            if (prob.data[:, :self.nc].sum(1) == 0).sum() != 0:  # in case of log(0)
                soft_weight = torch.FloatTensor(batch_size).fill_(0)
                soft_weight[prob[:, :self.nc].sum(1).data.cpu() == 0] = 1e-6
                soft_weight_var = Variable(soft_weight).cuda()
                loss = -((prob[:, :self.nc].sum(1) + soft_weight_var).log().mean())
            else:
                loss = -(prob[:, :self.nc].sum(1).log().mean())
            return loss
        if area == 'right':
            if (prob.data[:, self.nc:].sum(1) == 0).sum() != 0:  # in case of log(0)
                soft_weight = torch.FloatTensor(batch_size).fill_(0)
                soft_weight[prob[:, self.nc:].sum(1).data.cpu() == 0] = 1e-6
                soft_weight_var = Variable(soft_weight).cuda()
                loss = -((prob[:, self.nc:].sum(1) + soft_weight_var).log().mean())
            else:
                loss = -(prob[:, self.nc:].sum(1).log().mean())
            return loss

    def em_loss(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)
        prob_source = prob[:, :self.nc]
        prob_target = prob[:, self.nc:]
        prob_sum = prob_target + prob_source
        if (prob_sum.data.cpu() == 0).sum() != 0:  # in case of log(0)
            weight_sum = torch.FloatTensor(batch_size, self.nc).fill_(0)
            weight_sum[prob_sum.data.cpu() == 0] = 1e-6
            weight_sum = Variable(weight_sum).cuda()
            loss_sum = -(prob_sum + weight_sum).log().mul(prob_sum).sum(1).mean()
        else:
            loss_sum = -prob_sum.log().mul(prob_sum).sum(1).mean()

        return loss_sum

    def forward(self, logits, label, epoch):
        output1 = logits[0]
        output2 = logits[1]
        output_s1 = output1[:self.batch_size, :]
        output_s2 = output2[:self.batch_size, :]
        output_t1 = output1[self.batch_size:, :]
        output_t2 = output2[self.batch_size:, :]
        output_pt1 = F.softmax(output_t1)
        output_pt2 = F.softmax(output_t2)
        output_sts = torch.cat((output_s1, output_s2), dim=1)
        output_stt = torch.cat((output_t1, output_t2), dim=1)

        entropy_loss = - torch.mean(torch.log(torch.mean(output_pt1, 0)+1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_pt2, 0)+1e-6))
        loss1 = self.cls_loss(output_s1, label)
        loss2 = self.cls_loss(output_s2, label)
        # dis_loss = self.measure_dis(self.measure, output_pt1, output_pt2)
        domain_loss = self.st_loss(output_sts, area='left') + self.st_loss(output_stt, area='right')
        st_loss1 = self.cls_loss(output_sts, label)
        st_loss2 = self.cls_loss(output_sts, label + self.nc)
        st_cat_loss = 0.5 * st_loss1 + 0.5 * st_loss2
        em_loss = self.em_loss(output_stt)
        st_dom_loss = 0.5 * self.st_loss(output_stt, area='left') + 0.5 * self.st_loss(output_stt, area='right') + em_loss

        all_loss = loss1 + loss2 + 0.01 * entropy_loss
        # f_loss = loss1 + loss2 - self.eta * dis_loss + 0.01 * entropy_loss
        f_loss = loss1 + loss2 + 0.01 * entropy_loss + domain_loss
        # g_loss = dis_loss
        lam = 2 / (1 + math.exp(-1 * 10 * epoch / self.epochs)) - 1
        st_loss = st_cat_loss + lam * st_dom_loss
        # g_loss = dis_loss
        return all_loss, f_loss, st_loss


def cf_only(pretrained=True, args=None):
    model = Model(pretrained, args)
    loss_model = Loss(args)
    return model, loss_model
