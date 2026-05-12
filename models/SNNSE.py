import numpy as np
import torch
import torch.nn as nn
import collections
import  matplotlib.pyplot as plt
import torch.nn.functional as F

def attention_Weight(input, level):
    level -= 1
    inputMax = torch.max(input).item()
    inputMin = torch.min(input).item()

    out = torch.zeros_like(input)

    inputRange = (inputMax - inputMin)/level

    for i in range(level):
        out += (input > (inputMin + (inputRange * (i+1))/2)).float()

    return out

def attention_WeightS(input, level):
    level -= 1
    inputMax = torch.max(input).item()
    inputMin = torch.min(input).item()

    out = torch.zeros_like(input)

    inputRange = (inputMax - inputMin)/level

    for i in range(level):
        out += (input > (inputMin + (inputRange * (i+1))/2)).float()

    out = out/out.max() * inputMax
    return out

def attention_WeightB(input, bite):
    level = int(2 ** bite)
    inputMax = torch.max(input).item()
    inputMin = torch.min(input).item()

    # temp = np.linspace(0, 2, int(2 ** bite - 1))
    # tempMin = 2
    # for i in temp:
    #     if abs(i - input)
    out = torch.zeros_like(input)

    inputRange = (inputMax - inputMin)/level

    for i in range(level):
        out += (input > (inputMin + (inputRange * (i+1))/2)).float()

    out = out / level * 1.5

    return out

def collect_Weight(weight, i):
    # tempWeight = attention_Weight(weight, 100)
    # unique, counts = np.unique(tempWeight.cpu().detach().numpy(), return_counts=True)
    # if i == 18955:
    if i == 115:
        plt.hist(weight.cpu().detach().numpy().flatten(), bins=50, alpha=0.5)
        plt.show()
    i += 1
    print(i)
    return i

    # collect = np.zeros([2, 1])
    # for i in tempArray[:,1,:,:].squeeze():
    #     if np.in1d(i, collect):
    #         index = np.where(np.isin(collect, i))
    #         collect[1, index] += 1
    #     elif collect[0, 0] == 0:
    #         collect[0, 0] = i
    #         collect[1, 0] += 1

def weights_Code(x, threshold, T, sign):
    membrane = 0.5 * threshold
    sum_spikes = 0

    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    train_shape = [x.shape[0], x.shape[1]]
    x = x.flatten(0, 1)
    train_shape.extend(x.shape[1:])
    x = x.reshape(train_shape)

    # integrate charges
    for dt in range(T):
        membrane = membrane + x[:, dt]
        if dt == 0:
            spike_train = torch.zeros(membrane.shape[:1] + torch.Size([T]) + membrane.shape[1:],
                                      device=membrane.device)

        spikes = membrane >= threshold
        membrane[spikes] = membrane[spikes] - threshold
        spikes = spikes.float()
        sum_spikes = sum_spikes + spikes

        ###signed spikes###
        if sign:
            inhibit = membrane <= -1e-3
            inhibit = inhibit & (sum_spikes > 0)
            membrane[inhibit] = membrane[inhibit] + threshold
            inhibit = inhibit.float()
            sum_spikes = sum_spikes - inhibit
        else:
            inhibit = 0

        spike_train[:, dt] = spikes - inhibit

    # spike_train = spike_train * threshold
    return spike_train

def loss_kld(inputs, targets):
    inputs = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')