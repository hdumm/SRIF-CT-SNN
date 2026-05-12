import torch
import torch.nn as nn
from .quant_layer import (QuantConv2d, QuantLinear, QuantReLU, first_conv, last_fc, QuantAvg)
from .spiking import Spiking, last_Spiking, IF
from .spikingT import SpikingT
from .SNNSE import attention_Weight, attention_WeightS, collect_Weight, attention_WeightB, loss_kld
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

fs = 200  # sampling frequency
channel = 22  # number of electrode
num_input = 1  # number of channel picture (for EEG signal is always : 1)
num_class = 4  # number of classes
signal_length = 512  # number of sample in each tarial
ratio = 5
inFeatures = 15330
size_class = 200

F1 = 10  # number of temporal filters
D = 3  # depth multiplier (number of spatial filters)
F2 = D * F1  # number of pointwise filters

kernel_size_1 = (1, round(fs / 2))
kernel_size_2 = (channel, 1)
kernel_size_3 = (1, round(fs / 8))
kernel_size_4 = (1, 1)

kernel_avgpool_1 = (1, 4)
kernel_avgpool_2 = (1, 8)
dropout_rate = 0.01

ks0 = int(round((kernel_size_1[0] - 1) / 2))
ks1 = int(round((kernel_size_1[1] - 1) / 2))
kernel_padding_1 = (ks0, ks1 - 1)
ks0 = int(round((kernel_size_3[0] - 1) / 2))
ks1 = int(round((kernel_size_3[1] - 1) / 2))
kernel_padding_3 = (ks0, ks1)

class EEGNet(nn.Module):
    def __init__(self, nb_classes: int, float=False, T=15) -> None:
        super(EEGNet, self).__init__()
        if float:
            self.layer1 = Dummy(nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                                              nn.BatchNorm2d(F1),
                                              nn.Conv2d(F1, F1 * D, kernel_size_2, groups=F1),
                                              nn.BatchNorm2d(F1 * D),
                                              nn.ReLU(inplace=True)
                                              ))
            self.layer2 = Dummy(nn.Sequential(nn.Conv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                                              nn.Conv2d(F2, F2, kernel_size_4),
                                              nn.BatchNorm2d(F1 * D),
                                              nn.ReLU(inplace=True),
                                              ))
            self.layer3 = Dummy(nn.Sequential(nn.Linear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              nn.ReLU(inplace=True)))
            self.layer4 = Dummy(nn.Linear(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))
        else:
            self.layer1 = Dummy(nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                                              nn.BatchNorm2d(F1),
                                              QuantConv2d(F1, F2, kernel_size_2, groups=F1),
                                              nn.BatchNorm2d(F2),
                                              QuantReLU(inplace=True),
                                              ))
            self.layer2 = Dummy(
                nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                              QuantConv2d(F2, F2, kernel_size_4),
                              nn.BatchNorm2d(F2),
                              QuantReLU(inplace=True),
                              ))
            self.layer3 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              QuantReLU(inplace=True)))
            self.layer4 = Dummy(last_fc(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.flat(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        a1 = torch.flatten(x1)
        a2 = torch.flatten(x2)
        a3 = torch.flatten(x3)
        a4 = torch.flatten(x4)
        a5 = torch.flatten(x5)

        a = torch.cat((a1,a2), dim=0).cpu()
        # torch.save(a,"EEGNetAct2a.pt")
        return x5


    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()

class EEGNet_CT(nn.Module):
    def __init__(self, nb_classes: int, float=False, T=15) -> None:
        super(EEGNet_CT, self).__init__()
        if float:
            # ========================================================================================================
            # CNN
            # ========================================================================================================
            self.layer1 = Dummy(nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                                              nn.BatchNorm2d(F1),
                                              nn.Conv2d(F1, F1 * D, kernel_size_2, groups=F1),
                                              nn.BatchNorm2d(F1 * D),
                                              nn.ReLU(inplace=True)
                                              ))
            self.layer2 = Dummy(nn.Sequential(nn.Conv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                                              nn.Conv2d(F2, F2, kernel_size_4),
                                              nn.BatchNorm2d(F1 * D),
                                              nn.ReLU(inplace=True),
                                              ))
            self.layer3 = Dummy(nn.Sequential(nn.Linear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              nn.ReLU(inplace=True)))
            self.layer4 = Dummy(nn.Linear(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))
            # ========================================================================================================
            # SNN网络
            # ========================================================================================================
            self.slayer1 = SpikingT(
                nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                              nn.BatchNorm2d(F1),
                              QuantConv2d(F1, F2, kernel_size_2, groups=F1),
                              nn.BatchNorm2d(F2),
                              IF(),
                              ), T, 4)
            self.slayer2 = SpikingT(
                nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                              QuantConv2d(F2, F2, kernel_size_4),
                              nn.BatchNorm2d(F2),
                              IF(),
                              ), T, 3)
            self.slayer3 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
                                                  nn.BatchNorm1d(size_class),
                                                  IF()), T, 2)
            self.slayer4 = last_Spiking(last_fc(size_class, num_class), T)
            self.sflat = Dummy(nn.Flatten(2))

            self.slayer1.is_first = True
            self.slayer3.is_classer = True
        else:
            # ========================================================================================================
            # 量化CNN
            # ========================================================================================================
            self.layer1 = Dummy(nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                                              nn.BatchNorm2d(F1),
                                              QuantConv2d(F1, F2, kernel_size_2, groups=F1),
                                              nn.BatchNorm2d(F2),
                                              QuantReLU(inplace=True),
                                              ))
            self.layer2 = Dummy(
                nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                              QuantConv2d(F2, F2, kernel_size_4),
                              nn.BatchNorm2d(F2),
                              QuantReLU(inplace=True),
                              ))
            self.layer3 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
                                              nn.BatchNorm1d(size_class),
                                              QuantReLU(inplace=True)))
            self.layer4 = Dummy(last_fc(size_class, num_class))
            self.flat = Dummy(nn.Flatten(1))
            # ========================================================================================================
            # SNN网络
            # ========================================================================================================
            self.slayer1 = SpikingT(
                nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                              nn.BatchNorm2d(F1),
                              QuantConv2d(F1, F2, kernel_size_2, groups=F1),
                              nn.BatchNorm2d(F2),
                              IF(),
                              ), T, 4)
            self.slayer2 = SpikingT(
                nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                              QuantConv2d(F2, F2, kernel_size_4),
                              nn.BatchNorm2d(F2),
                              IF(),
                              ), T, 3)
            self.slayer3 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
                                                  nn.BatchNorm1d(size_class),
                                                  IF()), T, 2)
            self.slayer4 = last_Spiking(last_fc(size_class, num_class), T)
            self.sflat = Dummy(nn.Flatten(2))

            self.slayer1.is_first = True
            self.slayer3.is_classer = True
            # ========================================================================================================
            # 原始CNN网络
            # ========================================================================================================
            # self.layer1 = Dummy(nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1),
            #                                   nn.BatchNorm2d(F1),
            #                                   nn.Conv2d(F1, F1 * D, kernel_size_2, groups=F1),
            #                                   nn.BatchNorm2d(F1 * D),
            #                                   nn.ReLU(inplace=True)
            #                                   ))
            # self.layer2 = Dummy(nn.Sequential(nn.Conv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
            #                                   nn.Conv2d(F2, F2, kernel_size_4),
            #                                   nn.BatchNorm2d(F1 * D),
            #                                   nn.ReLU(inplace=True),
            #                                   ))
            # self.layer3 = Dummy(nn.Sequential(nn.Linear(inFeatures, size_class),
            #                                   nn.BatchNorm1d(size_class),
            #                                   nn.ReLU(inplace=True)))
            # self.layer4 = Dummy(nn.Linear(size_class, num_class))
            # self.flat = Dummy(nn.Flatten(1))
            # ========================================================================================================
            # 脉冲发射统计网络
            # ========================================================================================================
            # self.tlayer1 = SpikingT(nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
            #                                      nn.BatchNorm2d(F1),
            #                                      IF(),
            #                                      ), T, 2)
            # self.tlayer2 = SpikingT(nn.Sequential(QuantConv2d(F1, F2, kernel_size_2, groups=F1),
            #                                       nn.BatchNorm2d(F2),
            #                                       IF(),
            #                                       ), T, 2)
            # self.tlayer3 = SpikingT(nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
            #                                       nn.BatchNorm2d(F2),
            #                                       IF(),
            #                                       ), T, 2)
            # self.tlayer4 = SpikingT(nn.Sequential(QuantConv2d(F2, F2, kernel_size_4),
            #                                       nn.BatchNorm2d(F2),
            #                                       IF(),
            #                                       ), T, 2)
            # self.tlayer5 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
            #                                      nn.BatchNorm1d(size_class),
            #                                      IF()), T, 2)
            # self.tlayer6 = last_Spiking(last_fc(size_class, num_class), T)
            # self.tflat = Dummy(nn.Flatten(2))
            #
            # self.tlayer1.is_first = True
            # self.tlayer5.is_classer = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.detach().clone()
        t = x.detach().clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flat(x)
        x = self.layer3(x)
        x = self.layer4(x)

        self.slayer1.block[0] = self.layer1.block[0]
        self.slayer1.block[1] = self.layer1.block[1]
        self.slayer1.block[2] = self.layer1.block[2]
        self.slayer1.block[3] = self.layer1.block[3]

        self.slayer2.block[0] = self.layer2.block[0]
        self.slayer2.block[1] = self.layer2.block[1]
        self.slayer2.block[2] = self.layer2.block[2]

        self.slayer3.block[0] = self.layer3.block[0]
        self.slayer3.block[1] = self.layer3.block[1]

        # self.slayer1.block[4] = self.layer1.block[4]
        # self.slayer2.block[3] = self.layer2.block[3]
        # self.slayer3.block[2] = self.layer3.block[2]

        self.slayer4.block = self.layer4.block

        s1 = self.slayer1(s)
        s2 = self.slayer2(s1)
        s3 = self.sflat(s2)
        s4 = self.slayer3(s3)
        s5 = self.slayer4(s4)

        # t1 = self.tlayer1(t)
        # t2 = self.tlayer2(t1)
        # t3 = self.tlayer3(t2)
        # t4 = self.tlayer4(t3)
        # t5 = self.tflat(t4)
        # t6 = self.tlayer5(t5)
        # t7 = self.tlayer6(t6)

        # return x
        return s5


    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()


class EEGNet_E(nn.Module):
    def __init__(self, float=False, T=15) -> None:
        super(EEGNet_E, self).__init__()
        # # ========================================================================================================
        # # 量化CNN
        # # ========================================================================================================
        # self.layer1 = Dummy(nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
        #                                   nn.BatchNorm2d(F1),
        #                                   QuantConv2d(F1, F2, kernel_size_2, groups=F1),
        #                                   nn.BatchNorm2d(F2),
        #                                   QuantReLU(inplace=True),
        #                                   ))
        # self.layer2 = Dummy(
        #     nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
        #                   QuantConv2d(F2, F2, kernel_size_4),
        #                   nn.BatchNorm2d(F2),
        #                   QuantReLU(inplace=True),
        #                   ))
        # self.layer3 = Dummy(nn.Sequential(QuantLinear(inFeatures, size_class),
        #                                   nn.BatchNorm1d(size_class),
        #                                   QuantReLU(inplace=True)))
        # self.layer4 = Dummy(last_fc(size_class, num_class))
        # self.flat = Dummy(nn.Flatten(1))

        # self.slayer1 = Dummy(
        #     nn.Sequential(nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1),
        #                   nn.BatchNorm2d(F1),
        #                   nn.Conv2d(F1, F2, kernel_size_2, groups=F1),
        #                   nn.BatchNorm2d(F2),
        #                   nn.ReLU(),
        #                   ))
        # self.slayer2 = Dummy(
        #     nn.Sequential(nn.Conv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
        #                   nn.Conv2d(F2, F2, kernel_size_4),
        #                   nn.BatchNorm2d(F2),
        #                   nn.ReLU(),
        #                   ))
        # # self.slayer3 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
        # #                                       nn.BatchNorm1d(size_class),
        # #                                       IF()), T, 2)
        # self.slayer4 = Dummy(last_fc(size_class, num_class))
        # self.sflat = Dummy(nn.Flatten(2))

        # # ========================================================================================================
        # # SNN网络
        # # ========================================================================================================
        self.slayer1 = SpikingT(
            nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
                          nn.BatchNorm2d(F1),
                          QuantConv2d(F1, F2, kernel_size_2, groups=F1),
                          nn.BatchNorm2d(F2),
                          IF(),
                          ), T, 4)
        self.slayer2 = SpikingT(
            nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
                          QuantConv2d(F2, F2, kernel_size_4),
                          nn.BatchNorm2d(F2),
                          IF(),
                          ), T, 3)
        # self.slayer3 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
        #                                       nn.BatchNorm1d(size_class),
        #                                       IF()), T, 2)
        self.slayer4 = last_Spiking(last_fc(size_class, num_class), T)
        self.sflat = Dummy(nn.Flatten(2))

        # self.slayer1.is_first = True
        # self.slayer3.is_classer = True

        # self.ss11 = SpikingT(nn.Sequential(first_conv(num_input, F1, kernel_size_1, padding=kernel_padding_1),
        #                   nn.BatchNorm2d(F1),
        #                   IF(),
        #                   ), T, 2)
        # self.ss12 = SpikingT(nn.Sequential(QuantConv2d(F1, F2, kernel_size_2, groups=F1),
        #                   nn.BatchNorm2d(F2),
        #                   IF(),
        #                   ), T, 2)
        #
        # self.ss21 = SpikingT(
        #     nn.Sequential(QuantConv2d(F2, F2, kernel_size_3, padding=kernel_padding_3, groups=F2),
        #                   IF(),
        #                   ), T, 1)
        # self.ss22 = SpikingT(nn.Sequential(QuantConv2d(F2, F2, kernel_size_4),
        #                   nn.BatchNorm2d(F2),
        #                   IF(),
        #                   ), T, 2)
        # self.ss3 = SpikingT(nn.Sequential(QuantLinear(inFeatures, size_class),
        #                                       nn.BatchNorm1d(size_class),
        #                                       IF()), T, 2)
        # self.ss4 = last_Spiking(last_fc(size_class, num_class), T)
        # self.sflat = Dummy(nn.Flatten(2))
        #
        # self.ss11.is_first = True
        # self.ss3.is_classer = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.detach().clone()
        t = x.detach().clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flat(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # self.slayer1.block[0] = self.layer1.block[0]
        # self.slayer1.block[1] = self.layer1.block[1]
        # self.slayer1.block[2] = self.layer1.block[2]
        # self.slayer1.block[3] = self.layer1.block[3]

        self.ss11.block[0] = self.layer1.block[0]
        self.ss11.block[1] = self.layer1.block[1]
        self.ss12.block[0] = self.layer1.block[2]
        self.ss12.block[1] = self.layer1.block[3]

        # self.slayer2.block[0] = self.layer2.block[0]
        # self.slayer2.block[1] = self.layer2.block[1]
        # self.slayer2.block[2] = self.layer2.block[2]

        self.ss21.block[0] = self.layer2.block[0]
        self.ss22.block[0] = self.layer2.block[1]
        self.ss22.block[1] = self.layer2.block[2]

        # self.slayer3.block[0] = self.layer3.block[0]
        # self.slayer3.block[1] = self.layer3.block[1]

        self.ss3.block[0] = self.layer3.block[0]
        self.ss3.block[1] = self.layer3.block[1]

        # self.slayer4.block = self.layer4.block
        self.ss4.block = self.layer4.block

        s11 = self.ss11(s)
        s12 = self.ss12(s11)
        s21 = self.ss21(s12)
        s22 = self.ss22(s21)
        s3 = self.sflat(s22)
        s4 = self.ss3(s3)
        s5 = self.ss4(s4)

        r11 = torch.sum(s11)/2
        r12 = torch.sum(s12)/2
        r21 = torch.sum(s21)/2
        r22 = torch.sum(s22)/2
        r3 = torch.sum(s3)/2
        r4 = torch.sum(s4)/2
        r5 = torch.sum(s5)/2

        # t1 = self.tlayer1(t)
        # t2 = self.tlayer2(t1)
        # t3 = self.tlayer3(t2)
        # t4 = self.tlayer4(t3)
        # t5 = self.tflat(t4)
        # t6 = self.tlayer5(t5)
        # t7 = self.tlayer6(t6)

        return s5


    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()