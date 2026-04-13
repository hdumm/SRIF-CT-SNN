import torch
import torch.nn as nn
from .quant_layer import (QuantConv2d, QuantLinear, QuantReLU, first_conv, last_fc, QuantAvg)
from .spiking import Spiking, last_Spiking, IF
from .spikingT import SpikingT
from .utilsSelf import attention_Weight, attention_WeightS, collect_Weight, attention_WeightB, weights_Code, weightsTrans
from torch.nn import functional as F

class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        # print("x.size:{}", x.shape)
        if self.idem:
            return x
        return self.block(x)

fs = 250  # sampling frequency
channel = 22  # number of electrode
num_input = 1  # number of channel picture (for EEG signal is always : 1)
num_class = 4  # number of classes
signal_length = 512  # number of sample in each tarial
ratio = 5
inFeatures = 19520
size_class = 200

F1 = 40  # number of temporal filters
F2 = 40
F3 = F2 * 1
F4 = F3 * 2

kernel_size_11 = (1, 25)
kernel_size_12 = (channel, 1)
kernel_size_21 = (1, 75)
kernel_size_31 = (1, 50)
kernel_size_41 = (1, 10)

# float参数的取值决定了模型中使用的卷积层、线性层和激活函数的类型，影响模型的计算方式和性能特性。
# 当float为True时，使用的是标准的浮点数实现
# 当float为False时，使用量化实现，将浮点数作为整形进行计算
class ShallowConvNet(nn.Module):
    def __init__(self, float=False, T: int = 3) -> None:
        super(ShallowConvNet, self).__init__()
        if float:
            self.layer1 = Dummy(nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_11),
                                                nn.Conv2d(F1, F1, kernel_size_12, groups=F1),
                                                nn.BatchNorm2d(F1),
                                                nn.ReLU(inplace=True)
                                              ))

            self.layer2 = Dummy(nn.Sequential(nn.Linear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              nn.ReLU(inplace=True)))
            self.layer3 = Dummy(nn.Linear(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))
        else:
            self.layer1 = Dummy(nn.Sequential(first_conv(num_input, F1, kernel_size_11),
                                                QuantConv2d(F1, F1, kernel_size_12, groups=F1),
                                                nn.BatchNorm2d(F1),
                                                QuantReLU(inplace=True)
                                              ))

            self.layer2 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              QuantReLU(inplace=True)))
            self.layer3 = Dummy(last_fc(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))

            # self.layer5 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
            #                                   nn.BatchNorm1d(size_class),
            #                                   QuantReLU(inplace=True)))
            # self.layer6 = Dummy(nn.Linear(size_class, num_class))
            # self.flat = Dummy(nn.Flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.flat(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        a = torch.flatten(x1).cpu()
        torch.save(a, 'SCNAct2a.pt')
        return x4

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()

class ShallowConvNet_CT(nn.Module):
    def __init__(self, float=False, T: int = 3) -> None:
        super(ShallowConvNet_CT, self).__init__()
        if float:
            self.layer1 = Dummy(nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_11),
                                              nn.Conv2d(F1, F1, kernel_size_12, groups=F1),
                                              nn.BatchNorm2d(F1),
                                              nn.ReLU(inplace=True)
                                              ))

            self.layer2 = Dummy(nn.Sequential(nn.Linear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              nn.ReLU(inplace=True)))
            self.layer3 = Dummy(nn.Linear(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))

            self.slayer1 = SpikingT(nn.Sequential(first_conv(num_input, F1, kernel_size_11),
                                                  QuantConv2d(F1, F1, kernel_size_12, groups=F1),
                                                  nn.BatchNorm2d(F1),
                                                  IF()), T, 3)
            self.slayer2 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
                                                  nn.BatchNorm1d(size_class),
                                                  IF()), T, 2)
            self.slayer3 = last_Spiking(last_fc(size_class, num_class), T)
            self.sflat = Dummy(nn.Flatten(2))

            self.slayer1.is_first = True
            self.slayer2.is_classer = True
        else:
            self.layer1 = Dummy(nn.Sequential(first_conv(num_input, F1, kernel_size_11),
                                                QuantConv2d(F1, F1, kernel_size_12, groups=F1),
                                                nn.BatchNorm2d(F1),
                                                QuantReLU(inplace=True)
                                              ))

            self.layer2 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              QuantReLU(inplace=True)))
            self.layer3 = Dummy(last_fc(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))

            self.slayer1 = SpikingT(nn.Sequential(first_conv(num_input, F1, kernel_size_11),
                                              QuantConv2d(F1, F1, kernel_size_12, groups=F1),
                                              nn.BatchNorm2d(F1),
                                              IF()), T, 3)
            self.slayer2 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              IF()), T, 2)
            self.slayer3 = last_Spiking(last_fc(size_class, num_class), T)
            self.sflat = Dummy(nn.Flatten(2))

            self.slayer1.is_first = True
            self.slayer2.is_classer = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.detach().clone()
        x1 = self.layer1(x)
        x2 = self.flat(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)


        self.slayer1.block[0] = self.layer1.block[0]
        self.slayer1.block[1] = self.layer1.block[1]
        self.slayer1.block[2] = self.layer1.block[2]

        self.slayer2.block[0] = self.layer2.block[0]
        self.slayer2.block[1] = self.layer2.block[1]

        # self.slayer1.block[3] = self.layer1.block[3]
        # self.slayer2.block[2] = self.layer2.block[2]
        # self.slayer3.block[2] = self.layer3.block[2]
        # self.slayer4.block[2] = self.layer4.block[2]
        # self.slayer5.block[2] = self.layer5.block[2]

        self.slayer3.block = self.layer3.block

        s1 = self.slayer1(s)
        s2 = self.sflat(s1)
        s3 = self.slayer2(s2)
        s4 = self.slayer3(s3)
        a = torch.flatten(torch.sum(s1, dim=1)/15).cpu()
        torch.save(a, 'SCNAct2asRif2.pt')
        # return x4
        return s4

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()

class ShallowConvNet_E(nn.Module):
    def __init__(self, float=False, T: int = 3) -> None:
        super(ShallowConvNet_E, self).__init__()
        # self.layer1 = Dummy(nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_11),
        #                                   nn.Conv2d(F1, F1, kernel_size_12, groups=F1),
        #                                   nn.BatchNorm2d(F1),
        #                                   nn.ReLU(inplace=True)
        #                                   ))
        # self.layer11 = Dummy(nn.Sequential(nn.MaxPool2d([1,75])))
        # self.layer3 = Dummy(last_fc(size_class, num_class))
        # self.flat = Dummy(nn.Flatten(1))

        # self.layer2 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
        #                                   nn.BatchNorm1d(size_class),
        #                                   QuantReLU(inplace=True)))

        self.layer1 = SpikingT(nn.Sequential(first_conv(num_input, F1, kernel_size_11),
                                          QuantConv2d(F1, F1, kernel_size_12, groups=F1),
                                          nn.BatchNorm2d(F1),
                                          IF()
                                          ), T, 3)
        self.layer11 = Dummy(nn.Sequential(nn.MaxPool2d([1,75])))
        self.layer3 = Dummy(last_Spiking(size_class, num_class))
        self.flat = Dummy(nn.Flatten(1))

        # self.slayer1 = SpikingT(nn.Sequential(first_conv(num_input, F1, kernel_size_11),
        #                                       IF()), T, 1)
        # self.slayer11 = SpikingT(nn.Sequential(QuantConv2d(F1, F1, kernel_size_12, groups=F1),
        #                                       nn.BatchNorm2d(F1),
        #                                       IF()), T, 2)
        # self.slayer2 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
        #                                       nn.BatchNorm1d(size_class),
        #                                       IF()), T, 2)
        # self.slayer3 = last_Spiking(last_fc(size_class, num_class), T)
        # self.sflat = Dummy(nn.Flatten(2))
        #
        # self.slayer1.is_first = True
        # self.slayer2.is_classer = True
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.detach().clone()
        x1 = self.layer1(x)
        x11 = self.layer11(x1)
        x2 = self.flat(x11)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)


        # self.slayer1.block[0] = self.layer1.block[0]
        #
        # self.slayer11.block[0] = self.layer1.block[1]
        # self.slayer11.block[1] = self.layer1.block[2]
        #
        # self.slayer2.block[0] = self.layer2.block[0]
        # self.slayer2.block[1] = self.layer2.block[1]
        #
        # # self.slayer1.block[3] = self.layer1.block[3]
        # # self.slayer2.block[2] = self.layer2.block[2]
        # # self.slayer3.block[2] = self.layer3.block[2]
        # # self.slayer4.block[2] = self.layer4.block[2]
        # # self.slayer5.block[2] = self.layer5.block[2]
        #
        # self.slayer3.block = self.layer3.block
        #
        # s1 = self.slayer1(s)
        # s11 = self.slayer11(s1)
        # s2 = self.sflat(s11)
        # s3 = self.slayer2(s2)
        # s4 = self.slayer3(s3)
        #
        # r1 = torch.sum(s1)/2
        # r11 = torch.sum(s11)/2
        # r2 = torch.sum(s2)/2
        # r3 = torch.sum(s3)/2
        # r4 = torch.sum(s4)/2
        # # return x4
        return x4

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()