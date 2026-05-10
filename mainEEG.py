import argparse
import os
import time
import shutil
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from models import *
from models import loaddata
# from models.CKA import cka
# from tools.BCIIV2b_Process import get_data
from scipy.io import loadmat

from thop import profile
from thop import clever_format

# 创建命令行解析器
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')

best_prec = 0
# 解析命令赋值信息
args = parser.parse_args()

# f: 256 1snn
RANDOM_SEED = 3  # any random number


def set_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
# set_seed(RANDOM_SEED)

def main():

    global args, best_prec
    use_gpu = torch.cuda.is_available()
    print(args.device)
    print('=> Building model...')
    model = None

    # ===========================================================================================
    # args.arch = 'EEG'
    # args.init = "result/EEG_32bit/model_best.pth.tar"
    # args.init = "result/EEG_4bit/model_best.pth.tar"
    # args.init = "result/EEG_4bit/q/model_best.pth.tar"
    # ===========================================================================================
    # args.arch = 'DeepConvNet'
    # args.init = "result/DeepConvNet_32bit_ST/model_best.pth.tar"
    # args.init = "result/DeepConvNet_4bit_ST/model_best.pth.tar"
    # args.init = "result/DeepConvNet_4bit_CT/model_best.pth.tar"
    # ===========================================================================================
    args.arch = 'EEGNet'
    # args.arch = 'EEGNet_E'
    # args.init = "D:\C_Panel\code_replication\checkpoint\EEGNet_32bit_ST\model_best.pth.tar"
    # args.init = "D:\C_Panel\code_replication\checkpoint\EEGNet_4bit_ST\model_best.pth.tar"
    args.init = "D:\C_Panel\code_replication\checkpoint\EEGNet_4bit_CT\model_best.pth.tar"
    # ===========================================================================================
    # args.arch = 'ShallowConvNet'
    # args.arch = 'ShallowConvNet_E'
    # args.init = "D:\C_Panel\code_replication\checkpoint\ShallowConvNet_32bit_ST\model_best.pth.tar"
    # args.init = "D:\C_Panel\code_replication\checkpoint\ShallowConvNet_4bit_ST\model_best.pth.tar"
    # args.init = "D:\C_Panel\code_replication\checkpoint\ShallowConvNet_4bit_CT\model_best.pth.tar"
    # ===========================================================================================
    args.print_freq = 10
    args.batch_size = 20
    # args.trainType = 'ST'
    args.trainType = 'CT'
    # args.modelType = 'CNN'
    args.modelType = 'QCNN'
    args.trainSign = True
    # args.trainSign = False
    path1 = "D:\C_Panel\code_replication\dataset\pre_A01T.mat"
    path2 = "D:\C_Panel\code_replication\dataset\pre_A01E.mat"

    path = r'D:\Files_of_Graduate_student\Public_datasets\BCICIV_2b_mat\\'
    subject = 0
    # ===========================================================================================
    # python main.py --arch eeg --bit 32 --wd 5e-4
    if args.modelType == 'CNN':
        args.bit = 32
        args.weight_decay = 5e-4
        args.epochs = 180
        args.lr = 1e-2
    # ===========================================================================================
    if args.modelType == 'QCNN':
        args.bit = 4
        args.weight_decay = 3e-4
        # args.weight_decay = 3e-5
        # args.lr = 1e-3
        args.lr = 5e-3
        # args.lr = 1e-2
        # args.lr = 3e-2
        if args.trainType == 'ST':
            args.epochs = 50
        elif args.trainType == 'CT':
            # args.lr = 1e-2
            # args.lr = 1e-3
            args.lr = 5e-4
            # args.lr = 1e-5
            args.epochs = 80
    # ===========================================================================================
    # python main.py --arch eeg --bit 3 --wd 1e-4  --lr 4e-2 --init result/eeg_4bit/model_best.pth.tar
    # args.bit = 3
    # args.weight_decay = 3e-5
    # args.lr = 4e-2
    # args.epochs = 1
    # ===========================================================================================
    # python main.py --arch eeg --bit 2 --wd 3e-5  --lr 4e-2 --init result/eeg_3bit/model_best.pth.tar
    # args.bit = 2
    # args.weight_decay = 3e-5
    # args.lr = 4e-2
    # ===========================================================================================
    if args.trainSign == False:
        args.epochs = 1

    if use_gpu:
        # float = True if args.bit == 32 else False
        float = True
        if args.trainType == 'ST':
            if args.arch == 'EEGNet':
                model = EEGNet(float=float, nb_classes=4, T=2 ** args.bit - 1)
            elif args.arch == 'DeepConvNet':
                model = DeepConvNet(float=float, T=2 ** args.bit - 1)
            elif args.arch == 'ShallowConvNet':
                model = ShallowConvNet(float=float, T=2 ** args.bit - 1)
            else:
                print('Architecture not support!')
                return
        elif args.trainType == 'CT':
            if args.arch == 'EEGNet':
                model = EEGNet_CT(float=float, nb_classes=4, T=2 ** args.bit - 1)
            elif args.arch == 'EEGNet_E':
                model = EEGNet_E(float=float, T=2 ** args.bit - 1)
            elif args.arch == 'DeepConvNet':
                model = DeepConvNet_CT(float=float, T=2 ** args.bit - 1)
            elif args.arch == 'ShallowConvNet':
                model = ShallowConvNet_CT(float=float, T=2 ** args.bit - 1)
            elif args.arch == 'ShallowConvNet_E':
                model = ShallowConvNet_E(float=float, T=2 ** args.bit - 1)
            else:
                print('Architecture not support!')
                return
        if not float:
            for m in model.modules():
                # Ouroboros-------determine quantization
                # Ouroboros-------确定量化
                # APoT quantization for weights, uniform quantization for activations
                # APoT量化权重，激活的均匀量化
                # Additive Powers-of-Two（APoT）加法二次幂量化，一种针对钟形和长尾分布的神经网络权重，有效的非均匀性量化方案。
                if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m,
                                                                                          QuantSeparableConv2D) or isinstance(
                        m, QuantDepthwiseConv2D):
                    # weight quantization, use APoT
                    m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
                if isinstance(m, QuantReLU):
                    # activation quantization, use uniform
                    m.act_grid_alpha = build_power_value(args.bit)
                    m.act_alq = act_quantization(b=args.bit, grid=m.act_grid_alpha, power=False)

        # DataParallel - 多GPU并行训练
        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+str(args.arch)+'_'+str(args.bit)+'bit'+'_'+str(args.trainType)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init)
            
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found!')
            exit()

    # BCI IV 2a ======================================================================================================
    print('=> loading EEG data...')
    data1 = loadmat(path1)
    data2 = loadmat(path2)
    eegData2aRaw = np.concatenate((data1['data_resampled'], data2['data_resampled']), axis=0)
    eegLabel2aRaw = np.concatenate((data1['label'], data2['label']), axis=1).squeeze()

    divideRate = 0.8
    dataSize = len(eegData2aRaw)

    eegData2aRaw = np.expand_dims(eegData2aRaw, axis=1).astype(np.float32)

    X_train = eegData2aRaw[:(int(divideRate * dataSize)), :, :, :]
    Y_train = eegLabel2aRaw[:(int(divideRate * dataSize))]
    X_test = eegData2aRaw[int(divideRate * dataSize):, :, :, :]
    Y_test = eegLabel2aRaw[int(divideRate * dataSize):]

    # data2b = get_data(path, subject, DataSet='BCI2a')
    # X_train = data2b[0].astype(np.float32)
    # Y_train = data2b[1].astype(np.uint8) - 1
    # X_test = data2b[3].astype(np.float32)
    # Y_test = data2b[4].astype(np.uint8) - 1

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    print("train batch size:", trainloader.batch_size,
          ", num of batch:", len(trainloader))

    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    testloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    print("train batch size:", testloader.batch_size,
          ", num of batch:", len(testloader))

    if args.evaluate:
        validate(testloader, model, criterion)
        # model.module.show_params()
        return

    # flops, params = profile(model, inputs=testloader)
    # print(flops, params)  # 1819066368.0 11689512.0
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)  # 1.819G 11.690M

    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # test_accuracies = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        if args.trainSign:
            train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)
        # test_accuracies.append(prec)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

    # plt.plot(range(1, args.epochs + 1), test_accuracies, label='ACC', marker='o', linestyle='-')
    # plt.xlabel('epoch')
    # plt.ylabel('ACC/%')
    # plt.xticks(range(1, args.epochs + 1))
    # plt.legend()
    # plt.grid(True)
    # plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % 2 == 0:
        #     model.module.show_params()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            # args.bit == 10:
            output = model(input)
            # args.bit == 4:
            # output, loss1, loss2 = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()