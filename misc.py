import torch
import numpy as np
import torch.nn.functional as F

# class A:
#     def p(self):
#         return 1
#
#     def a(self):
#         c = self.p()
#         print(c)
#
# aa=A()
# aa.a()
# a = torch.ones(2,3)  # s
# s = [a,a,a,a]
# b = 3*torch.ones(2,3)  # t
# t = [b,b,b,b]
# input = torch.cat((s, t), 0)
# a=torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
# # a=a.float()
# b=torch.tensor([[2,3,4],[4,5,6],[6,7,8],[8,9,10]])
# # c=F.softmax(a,dim=1)
# c=torch.cat((a,b),dim=1)
# d=c[:1,:]
# # print(c)
# print(c,d)
# b=torch.tensor([1,2,3,1])
# c=0
# c+=a.eq(b).cpu().sum()
# print(c.numpy())
# print('a: {:.4f}'.format(1.23234))
# def fun(a):
#     a.append(1)
#
# a=[1,2]
# fun(a)
# print(a)
# a = np.float64(3.22)
# print(a)
# print('a: {:.4f}'.format(a))
# s='resnet50'
# p=s.split('_')[0]
# print(p)
acc = 4.67777
print('test_{:.4f}.png'.format(acc))