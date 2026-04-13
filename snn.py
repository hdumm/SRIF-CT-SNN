import argparse
import os
import time
# import shutil

import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
from models import loaddata

from scipy.io import loadmat
import torchvision
import torchvision.transforms as transforms

from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Evaluation')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('-u', '--unsigned', dest='unsigned', action='store_true', help='use traditional unsigned spikes')

best_prec = 0
args = parser.parse_args()

RANDOM_SEED = 256 # any random number

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
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    print('=> Building model...')
    model = None

    # ===========================================================================================
    # 模拟命令行赋值内容：python snn.py --arch alex --bit 3 -e -u --init result/alex_3bit/model_best.pth.tar
    # args.arch = 'EEGNet'
    # args.init = "result/EEGNet_4bit/model_best.pth.tar"
    args.arch = 'EEG'
    # args.init = "result/EEG_32bit/model_best.pth.tar"
    args.init = "result/EEG_4bit/model_best.pth.tar"
    # args.init = "result/EEG_4bit/q/model_best.pth.tar"
    args.bit = 4
    args.evaluate = True
    args.unsigned = True
    args.batch_size = 1
    path1 = r'D:\Files_of_Graduate_student\Public_datasets\pre_2a\pre_A03T.mat'
    path2 = r'D:\Files_of_Graduate_student\Public_datasets\pre_2a\pre_A03E.mat'
    # ===========================================================================================

    float = True if args.bit == 32 else False

    if args.arch == 'EEGNet':
        model = S_EEGNet(T=2 ** args.bit - 1)
    elif args.arch == 'EEG':
        # model = EEG(float=float)
        model = S_EEG(T=2 ** args.bit - 1)
    elif args.arch == 'EEGNetIn':
        model = S_EEGNet(T=2 ** args.bit - 1)
    elif args.arch == 'EEGNetInSE':
        # model = EEGNetYY(float=float, nb_classes=2)
        model = S_EEGNetYY(T=2 ** args.bit - 1)
    elif args.arch == 'AlexNet':
        model = S_AlexNet(T=2 ** args.bit - 1)
    else:
        print('Architecture not support!')
        return
    if not float:
        for m in model.modules():
            #Ouroboros-------determine quantization
            #APoT quantization for weights, uniform quantization for activations
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantSeparableConv2D) or isinstance(m, QuantDepthwiseConv2D):
                #weight quantization, use APoT
                m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
            if isinstance(m, QuantReLU):
                #activation quantization, use uniform
                m.act_grid = build_power_value(args.bit)
                m.act_alq = act_quantization(b=args.bit, grid=m.act_grid, power=False)

    if args.init:

        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            if use_gpu:
                print('use gpu')
                checkpoint = torch.load(args.init)
                model = model.cuda()
                criterion = nn.CrossEntropyLoss().cuda()
                cudnn.benchmark = True
            else:
                print('use cpu')
                checkpoint = torch.load(args.init, map_location='cpu')
                criterion = nn.CrossEntropyLoss()

            #Remove DataParallel wrapper 'module'
            for name in list(checkpoint['state_dict'].keys()):
                checkpoint['state_dict'][name[7:]] = checkpoint['state_dict'].pop(name)

            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found !')
            exit()

    # ===========================================================================================
    # 将EEG数据填充进train_dataset和test_dataset
    # divideRate = 0.9
    # eegDataRaw, eegLabelRaw = loaddata.eegDataProcessing('f')
    #
    # for i in range(len(eegLabelRaw)):
    #     if eegLabelRaw[i] == -1:
    #         eegLabelRaw[i] = 0
    #
    # dataSize = len(eegLabelRaw)
    #
    # X_test = eegDataRaw[int(divideRate * dataSize):, :, :200] * 2
    # X_train = eegDataRaw[:(int(divideRate * dataSize)), :, :200] * 2
    # Y_test = eegLabelRaw[int(divideRate * dataSize):]
    # Y_train = eegLabelRaw[:(int(divideRate * dataSize))]

        # BCI IV 2a ======================================================================================================
        print('=> loading EEG data...')
        data1 = loadmat(path1)
        data2 = loadmat(path2)
        eegData2aRaw = np.concatenate((data1['data_resampled'], data2['data_resampled']), axis=0)
        eegLabel2aRaw = np.concatenate((data1['label'], data2['label']), axis=1).squeeze()

        divideRate = 0.8
        dataSize = len(eegData2aRaw)

        eegData2aRaw = np.expand_dims(eegData2aRaw, axis=1).astype(np.float32)

        # Electrode-by-electrode channel non-negativity
        # for i in range(eegData2aRaw.shape[0]):
        #     for j in range(eegData2aRaw.shape[2]):
        #         minD = eegData2aRaw[i, :, j, :].min()
        #         maxD = eegData2aRaw[i, :, j, :].max()
        #         # eegData2aRaw[i, :, j, :] = eegData2aRaw[i, :, j, :] - minD
        #         eegData2aRaw[i, :, j, :] = (eegData2aRaw[i, :, j, :] - minD)/(maxD - minD)

        # Overall non-negative
        # eegData2aRaw = (eegData2aRaw - eegData2aRaw.min())/(eegData2aRaw.max() - eegData2aRaw.min()) * 100

        # Delete negative
        # eegData2aRaw = eegData2aRaw / (eegData2aRaw.max() - eegData2aRaw.min()) * 255
        # neIndex = eegData2aRaw < 0
        # eegData2aRaw[neIndex] = 0

        # Envelopes
        # eegData2aRaw = abs(eegData2aRaw)

        X_train = eegData2aRaw[:(int(divideRate * dataSize)), :, :, :]
        Y_train = eegLabel2aRaw[:(int(divideRate * dataSize))]
        X_test = eegData2aRaw[int(divideRate * dataSize):, :, :, :]
        Y_test = eegLabel2aRaw[int(divideRate * dataSize):]

        # for i in range(dataSize):
        #     if eegLabel2aRaw[i] == 0:
        #         eegLabel2aRaw[i] = 0
        #     elif eegLabel2aRaw[i] == 1:
        #         eegLabel2aRaw[i] = 0
        #     elif eegLabel2aRaw[i] == 2:
        #         eegLabel2aRaw[i] = 1
        #     elif eegLabel2aRaw[i] == 3:
        #         eegLabel2aRaw[i] = 1

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

    if args.unsigned:
        unsigned_spikes(model)

    if args.evaluate:
        validate(testloader, model, criterion)
        # validateSelf(X_test, Y_test, model, criterion)
        return
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

def validateSelf(X_test, Y_test, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    test_size = 10

    end = time.time()
    with torch.no_grad():
        # for i, (input, target) in enumerate(val_loader):
        for i in range(int(len(Y_test)/test_size)):
            input, target = torch.from_numpy(X_test[i * test_size:(i + 1) * test_size, :, :]), torch.from_numpy(
                Y_test[i * test_size:(i + 1) * test_size])
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target.long())

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(Y_test), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(args.device), target.to(args.device)

            # compute output
            output = model(input)
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