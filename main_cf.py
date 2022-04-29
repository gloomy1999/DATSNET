import argparse
import models
import os
import random
import json
from utils import *
from torchvision import datasets, transforms
import torch
from dataset import *
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import scipy.io as sio

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description='Scene Classification')
parser.add_argument('--mode', type=str, default='train',
                    metavar='MODE', help='train or test')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cf_only', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--measure', '-m', metavar='MEASURE', default='L1',
                    help='the measure of distance between f1 and f2')
parser.add_argument('--source_data', '-src', metavar='SOURCE', dest='train_data',
                    help='source dataset')
parser.add_argument('--target_data', '-tar', metavar='TARGET', dest='val_data',
                    help='target dataset')
parser.add_argument('--batch_size', '-b', type=int, default=32,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=40,
                    metavar='N', help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '-wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('--layers', type=int, default=2, metavar='K',
                    help='numbers of layers for classifier')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='numbers of steps to repeat the generator update')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--save_path', '-s', metavar='PATH', default=None,
                    help='saving path')
parser.add_argument('--gpu', type=str, default='1', metavar='GPU_ID',
                    help='GPU id to use')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--version', '-v', type=str, default='0', metavar='Ver',
                    help='model version')
parser.add_argument('--acc', type=str, default='0', metavar='Ver',
                    help='model to test')
args = parser.parse_args()
best_prec = 0
val_acc = []
cls_loss, f_loss, g_loss = [], [], []
matplotlib.use('Agg')


def plot_graph(x_vals, y_vals, x_label, y_label, legend):
    for i in range(len(legend)):
        plt.xlabel(x_label[i])
        plt.ylabel(y_label[i])
        plt.plot(x_vals[i], y_vals[i])
        fileName = os.path.join(args.save_path, legend[i] + ".png")
        # fileName = args.save_path + '/' + legend[i] + ".png"
        plt.savefig(fileName)
        plt.close()


def main():
    global args, best_prec
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.iter_num = 1  # record loss every ${print_freq} times

    '''get number of classes'''
    args.pair = args.train_data[0] + '_' + args.val_data[0]
    with open('./data/nc.json', 'r') as f:
        nc_info = json.load(f)
        args.nc = nc_info[args.pair]

    '''set saving dir'''
    if args.save_path is None:
        args.save_path = os.path.join('./output', args.pair.upper(), args.version)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    '''random seed'''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
    torch.manual_seed(args.seed)

    '''load data'''
    train_path = os.path.join('/home/zzd/dataserver/zzd/TL', get_pair(args.pair.upper()), args.train_data)
    val_path = os.path.join('/home/zzd/dataserver/zzd/TL', get_pair(args.pair.upper()), args.val_data)
    data_transforms = {
        train_path: transforms.Compose([
            transforms.Scale(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        val_path: transforms.Compose([
            transforms.Scale(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
    dset_classes = dsets[train_path].classes
    print('\nclasses' + str(dset_classes) + '\n')

    train_loader = CVDataLoader()
    train_loader.initialize(dsets[train_path], dsets[val_path], args.batch_size, shuffle=True)
    train_dataset = train_loader.load_data()
    test_loader = CVDataLoader()
    test_loader.initialize(dsets[train_path], dsets[val_path], args.batch_size, shuffle=True)
    test_dataset = test_loader.load_data()

    '''model building'''
    model, criterion = models.__dict__[args.arch](pretrained=True, args=args)
    if args.gpu is not None:
        model = model.cuda()
        criterion = criterion.cuda()

    if args.mode == 'test':
        print("Testing! Arch:" + args.arch)
        path = os.path.join(args.save_path, args.arch + '_' + args.measure +
                            '_{}.pth'.format(args.acc))
        if args.acc == '0':
            print('No Model Here!')
            sys.exit()
        model.load_state_dict(torch.load(path)['state_dict'])
        model.eval()
        correct = 0
        correct2 = 0
        size = 0
        print(val_path)
        val_data_loader = torch.utils.data.DataLoader(dsets[val_path], batch_size=args.batch_size,
                                                      shuffle=False, num_workers=4, )
        target_labels = np.array([])
        pred_labels_f1 = np.array([])
        pred_labels_f2 = np.array([])
        print(len(val_data_loader))
        for data2, target2 in val_data_loader:
            data2, target2 = data2.cuda(), target2.cuda()
            data1, target1 = Variable(data2, volatile=True), Variable(target2)
            output1, output2 = model(data1)
            pred = output1.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target1.data).cpu().sum()
            pred_labels_f1 = np.append(pred_labels_f1, pred.cpu().numpy())
            target_labels = np.append(target_labels, target2.cpu().numpy())
            pred = output2.data.max(1)[1]  # get the index of the max log-probability
            pred_labels_f2 = np.append(pred_labels_f2, pred.cpu().numpy())
            k = target1.data.size()[0]
            correct2 += pred.eq(target1.data).cpu().sum()
            size += k
        acc1 = 1.0 * correct.numpy() / (1.0 * size)
        acc2 = 1.0 * correct2.numpy() / (1.0 * size)
        acc = max(acc1, acc2)
        print('Accuracy: {:.4f}'.format(acc))

        mat_path = os.path.join(args.save_path, 'test_{:.4f}.mat'.format(acc))
        if os.path.exists(mat_path):
            sys.exit()
        if correct2 > correct:
            sio.savemat(mat_path, {'pred': pred_labels_f2, 'target': target_labels})
        else:
            sio.savemat(mat_path, {'pred': pred_labels_f1, 'target': target_labels})
        class_names = val_data_loader.dataset.classes
        plot_confusion_matrix(mat_path, class_names, args.save_path, acc)
        sys.exit()

    '''set optimizer'''
    g_params = [v for k, v in model.named_parameters() if 'gen' in k]
    f_params = [v for k, v in model.named_parameters() if 'cls' in k]

    g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, g_params),
                                  args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    f_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, f_params),
                                  args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    '''training'''
    for epoch in range(args.start_epoch, args.epochs):
        train(train_dataset, model, criterion, g_optimizer, f_optimizer, epoch)

        prec = validate(test_dataset, model)
        val_acc.append(prec)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)

        # save model
        if epoch == args.epochs - 1 or (is_best and best_prec > 0.5):
            save_path = os.path.join(args.save_path, args.arch + '_' + args.measure +
                                     '_{:.4f}.pth'.format(prec))
            if epoch == args.epochs - 1:
                save_path = os.path.join(args.save_path, args.arch + '_' + args.measure +
                                         '_{:.4f}_last.pth'.format(prec))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
            }, save_path)
            print('saving!!!!')

        # plot graph
        x_vals, y_vals, x_label, y_label, legend = [], [cls_loss, f_loss, g_loss,val_acc], \
                                                   [], [], ["L_cls", "L_f", "L_cf","val_acc"]
        for i in range(len(legend)):
            if 'L_' in legend[i]:
                x_vals.append(range(1, args.iter_num))
                x_label.append('iter_num(*10)')
                y_label.append('loss')
            else:
                x_vals.append(range(1, epoch + 1 - args.start_epoch + 1))
                x_label.append('epoch_num')
                y_label.append('acc')
        plot_graph(x_vals, y_vals, x_label, y_label, legend)


def train(train_dataset, model, criterion, g_optimizer, f_optimizer, epoch):
    model.train()

    for batch_idx, data in enumerate(train_dataset):
        stage = 0
        if batch_idx * args.batch_size > 30000:
            break
        data1 = data['S']
        target1 = data['S_label']
        data2 = data['T']
        target2 = data['T_label']
        data1, target1 = data1.cuda(), target1.cuda()
        data2, target2 = data2.cuda(), target2.cuda()
        input = Variable(torch.cat((data1, data2), 0))
        target = Variable(target1)

        # Step A: train all networks to minimize the loss on source
        g_optimizer.zero_grad()
        f_optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target, epoch)
        current_loss = loss[stage]
        L_cls = current_loss
        current_loss.backward()
        g_optimizer.step()
        f_optimizer.step()
        stage += 1

        # Step B: train classifier to maximize discrepancy
        g_optimizer.zero_grad()
        f_optimizer.zero_grad()

        # block the G
        # for name, param in model.named_parameters():
        #     if 'gen' in name:
        #         param.requires_grad = False

        logits = model(input)
        loss = criterion(logits, target, epoch)
        current_loss = loss[stage]
        L_f = current_loss
        current_loss.backward(retain_graph=True)
        stage += 1

        # block the F
        for name, param in model.named_parameters():
            param.requires_grad = True
            if 'cls' in name:
                param.requires_grad = False

        # confuse the F,use the same cal graph
        g_optimizer.zero_grad()
        current_loss = loss[stage]
        L_cf = current_loss
        current_loss.backward()

        # update the F and G
        for p in model.parameters():
            p.requires_grad = True
        g_optimizer.step()
        f_optimizer.step()
        stage += 1

        for p in model.parameters():
            p.requires_grad = True

        # print result
        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] loss:'.format
                  (epoch, batch_idx, len(train_dataset)), end='')
            print('L_cls: {:.4f} L_f: {:.4f} L_cf: {:.4f}'.format
                  (L_cls.item(), L_f.item(), L_cf.item()))  # .data[0]
            args.iter_num = args.iter_num + 1
            cls_loss.append(L_cls.item())
            f_loss.append(L_f.item())
            g_loss.append(L_cf.item())


def validate(test_dataset, model):
    model.eval()
    correct = 0
    correct2 = 0
    size = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataset):
            if batch_idx * args.batch_size > 5000:
                break
            data2 = data['T']
            target2 = data['T_label']
            data2, target2 = data2.cuda(), target2.cuda()
            test_data, test_target = Variable(data2, volatile=True), Variable(target2)
            output1, output2 = model(test_data)
            pred = output1.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(test_target.data).cpu().sum()
            pred = output2.data.max(1)[1]
            k = test_target.data.size()[0]
            correct2 += pred.eq(test_target.data).cpu().sum()
            size += k
        f1_acc = 1.0 * correct.numpy() / (1.0 * size)
        f2_acc = 1.0 * correct2.numpy() / (1.0 * size)

    print('f1_acc: {:.4f} f2_acc: {:.4f}'.format(f1_acc, f2_acc))

    return max(f1_acc, f2_acc)


if __name__ == '__main__':
    main()
