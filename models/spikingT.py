import torch
import os
import torch.nn as nn
from .quant_layer import QuantLinear

def unsigned_spikes(model):
    for m in model.modules():
         if isinstance(m, SpikingT):
             m.sign = False

#####the spiking wrapper######
##### 脉冲神经元 ######

class SpikingT(nn.Module):
    def __init__(self, block, T, index):
        super(SpikingT, self).__init__()
        self.block = block
        self.T = T
        self.is_first = False
        self.is_classer = False
        self.idem = False
        self.sign = True
        self.index = index

    def forward(self, x):
        if self.idem:
            return x

        torch.set_printoptions(sci_mode=False)
        ### initialize membrane to half threshold
        ### 将膜初始化为半阈值
        threshold = self.block[self.index].act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0

        # prepare charges
        if self.is_first:
            x.unsqueeze_(1)
            x = x.repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        # integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:], device=membrane.device)

            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            # //////////////////////////
            spike_train_inhibit = spike_train

            ###signed spikes###
            if self.sign:
                if self.is_first:
                    inhibit = membrane <= -1e-3
                    inhibit = inhibit & (sum_spikes > 0)
                else:
                    inhibit = membrane <= -1e-3
                    inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            # Sign
            spike_train_inhibit[:, dt] = spikes - inhibit

        z_x_float = torch.sum(x.relu(), dim=1) / self.T / threshold
        z_x_spike = torch.sum(spike_train, dim=1) / self.T
        z_x_float_numSpike = (z_x_float * self.T).int()
        z_x_spike_numSpike = z_x_spike * self.T

        zsindex_pos = (z_x_float_numSpike - z_x_spike_numSpike).relu().int()
        # zsindex_neg = -1 * (z_x_spike_numSpike - z_x_float_numSpike).relu().int()

        zsindex_pos[zsindex_pos > self.T] = self.T

        enhence_spike = torch.zeros(spike_train.shape)

        if self.is_classer:
            init_enhence = torch.zeros(spike_train.shape[0], 1, spike_train.shape[2]).cuda()
        else:
            init_enhence = torch.zeros(spike_train.shape[0], 1, spike_train.shape[2], spike_train.shape[3],
                                       spike_train.shape[4]).cuda()
        index = torch.arange(self.T).to(zsindex_pos.device)
        zsindex = zsindex_pos.unsqueeze(1)

        for i in range(self.T):
            mask = zsindex > 1
            # Random???????
            temp_enhence = init_enhence.masked_fill(mask, 1)
            zsindex = zsindex - 1
            if i == 0:
                enhence_spike = temp_enhence
            else:
                enhence_spike = torch.cat((enhence_spike, temp_enhence), dim=1)

        spike_train_enhence = (spike_train + enhence_spike) * threshold
        spike_train_inhibit = spike_train_inhibit * threshold

        # tensorData = [z_x_float, torch.sum(spike_train_enhence/threshold, dim=1)/self.T, torch.sum(spike_train, dim=1)/self.T]
        # torch.save(tensorData, "tensorData.pt")

        # if not self.is_classer:
        #     similar_if = similar_c(torch.sum(x.relu(), dim=1) / self.T, torch.sum(spike_train, dim=1) / self.T)
        #     # similar_inhibt = similar_c(torch.sum(x.relu(), dim=1) / self.T,
        #     #                            torch.sum(spike_train_inhibit, dim=1) / self.T)
        #     similar_enhence = similar_c(torch.sum(x.relu(), dim=1) / self.T,
        #                                 torch.sum(spike_train_enhence, dim=1) / self.T)
        # else:
        #     similar_if = similar2(torch.sum(x.relu(), dim=1) / self.T,
        #                              torch.sum(spike_train, dim=1) / self.T)
        #     # similar_inhibt = similar2(torch.sum(x.relu(), dim=1) / self.T,
        #     #                           torch.sum(spike_train_inhibit, dim=1) / self.T)
        #     similar_enhence = similar2(torch.sum(x.relu(), dim=1) / self.T,
        #                                torch.sum(spike_train_enhence, dim=1) / self.T)
        # if self.is_first:
        #     print("Loss_if: {:2f}".format(similar_if))
        #     # print("Loss_inhibt: {:2f}".format(similar_inhibt))
        #     print("loss_enhence:{:2f}".format(similar_enhence))
        #     print("=====================================")
        #     save_similar(similar_if, similar_enhence)
        # # print("threshold:{:2f}".format(threshold))
        # #
        # if similar_inhibt > similar_enhence:
        #     # similar = similar_inhibt
        #     # save_similar(similar_if, similar)
        #     return spike_train_inhibit
        # else:
        #     #     similar = similar_enhence
        #     #     save_similar(similar_if, similar)
        #     return spike_train_enhence

        return spike_train_enhence
        # return spike_train_inhibit
        # return spike_train * threshold


def save_similar(similar_if, similar_):
    fdir = 'result/similar'
    fname = 'similar.pt'
    layer = 0
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    if os.path.isfile(fdir+'/'+fname):
        data = torch.load(fdir+'/'+fname)
        # if data['layer'] + 1 == 6:
        #     layer = 1
        # else:
        layer = data['layer'] + 1

        similar = data['similar']
        # similar.requires_grad = False
        # if similar[layer, 0] == 0.0:
        #     similar[layer, 0] = similar_if
        #     similar[layer, 1] = similar_
        # else:
        #     similar[layer, 0] = (similar[layer, 0] + similar_if)/2
        #     similar[layer, 1] = (similar[layer, 1] + similar_)/2
        # torch.save({'similar': similar, 'layer': layer}, fdir + '/' + fname)
    else:
        similar = torch.zeros([24,2])
    similar.requires_grad = False
    similar[layer, 0] = similar_if
    similar[layer, 1] = similar_
    torch.save({'similar': similar, 'layer': layer}, fdir+'/'+fname)

def similar_c(X, Y):
    matrix = torch.empty(X.size(0), X.size(1))
    for i in range(X.size(0)):
        for j in range(X.size(1)):
            x = X[i, j, :, :]
            y = Y[i, j, :, :]
            d = (x - y).abs()
            unit = torch.ones([x.shape[0], x.shape[1]], device=d.device)
            matrix[i, j] = 1 - (d.sum() / unit.sum() + 1e-5)
    similar = matrix.sum() / (matrix.size(0) * matrix.size(1))
    return similar

def similar2(X, Y):
    matrix = torch.empty(X.size(0), X.size(1))
    for i in range(X.size(0)):
        for j in range(X.size(1)):
            x = X
            y = Y
            d = (x - y).abs()
            unit = torch.ones([x.shape[0], x.shape[1]], device=d.device)
            matrix[i, j] = 1 - (d.sum() / unit.sum() + 1e-5)
    similar = matrix.sum() / (matrix.size(0) * matrix.size(1))
    return similar

def linear_CKA(X, Y):
    cka_matrix = torch.empty(X.size(0), X.size(1))
    for i in range(X.size(0)):
        for j in range(X.size(1)):
            if X.shape[2] == 1:
                x = X[i, j, :, :]
                y = Y[i, j, :, :]
            else:
                x = X[i, j, :, :].squeeze()
                y = Y[i, j, :, :].squeeze()
            hsic = linear_HSIC(x, y)
            var1 = torch.sqrt(linear_HSIC(x, x))
            var2 = torch.sqrt(linear_HSIC(y, y))
            cka_matrix[i, j] = hsic / ((var1 * var2) + 1e-5)
    cka = cka_matrix.sum() / (cka_matrix.size(0) * cka_matrix.size(1))
    return cka

def linear_HSIC( X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

class last_Spiking(nn.Module):
    def __init__(self, block, T):
        super(last_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        #integrate charges
        return x.sum(dim=1)

class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x

    def show_params(self):
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold activation alpha: {:2f}'.format(act_alpha))

    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)

class group_Spiking(nn.Module):
    def __init__(self, block, T, num, bit):
        super(group_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.num = num
        self.bit = bit
        self.idem = False

    def forward(self, x, y):
        spike = 7
        out = torch.zeros_like(x)
        level = torch.arange(0, self.bit) * spike
        y = y / (torch.max(y) - torch.min(y)) * spike * self.bit
        for i in range((torch.max(x) - torch.min(x)).int() + 1):
            out[:, :, i, :] += level[abs(torch.ones_like(level) * y[i] - level).argmin(0)]

        return out