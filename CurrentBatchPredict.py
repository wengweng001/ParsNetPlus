import dataprep_weakly
import copy
import pickle
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *

import warnings
warnings.filterwarnings('ignore')


def GenTrain(net, input1, spc, criterionGen, lrGen, acm, runs, epoch, device):
    # input: input data type 0-original data, 1-augment data, 2-unlabeldata
    print('generative training...')

    # shuffle the data
    nData = input1.shape[0]
    shuffled_indices = torch.randperm(nData)

    input1 = input1.to(device)

    pseu_idx = []
    pseu_er = []
    for i in range(nData):
        # load data

        indices = shuffled_indices[i:(i + 1)]

        data1 = input1[indices]

        # grow/prune
        if flexible:
            grow, prune = spc.update_gen(net, acm, data1, epoch)

            if grow and spc.count >= 1:
                # grow nodes
                print('{}nodes+{}'.format(net.nNodes,acm.k))
                net.nodeGrowing(nNewNode=1,device=device)
                if epoch<0:
                    net = pickle.loads(pickle.dumps(net))
                net = copy.deepcopy(net)

                # change si buffer
                if si:
                    newrow = torch.zeros(1,20).to(device)
                    newcol = torch.zeros(3,1).to(device)
                    if runs>0:
                        p_old['linear__weight']       = (torch.cat((p_old['linear__weight'],newrow),0)).detach()
                        p_old['linear__bias']         = (torch.cat((p_old['linear__bias'],torch.zeros(1).to(device)), 0)).detach()
                        p_old['linearOutput__weight'] = (torch.cat((p_old['linearOutput__weight'],newcol), 1)).detach()
                        W['linear__weight']       = (torch.cat((W['linear__weight'],newrow),0)).detach()
                        W['linear__bias']         = (torch.cat((W['linear__bias'],torch.zeros(1).to(device)), 0)).detach()
                        W['linearOutput__weight'] = (torch.cat((W['linearOutput__weight'],newcol), 1)).detach()
                    for n,p in net.network.named_buffers():
                        if 'si_init' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,newrow),0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,torch.zeros(1).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,newcol), 1)).detach())
                        elif 'si_optim' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,newrow),0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,torch.zeros(1).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,newcol), 1)).detach())
                        elif 'si_omega' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,newrow),0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,torch.zeros(1).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n),(torch.cat((p,newcol), 1)).detach())

            if prune and not grow and spc.count >= 10:
                # prune node
                Es = sum([acm.wm[i] * net.network.activation(net.network.linear(acm.center[i])) for i in range(acm.k)])
                pruneIdx = findLeastSignificantNode(Es)
                if pruneIdx is not None:
                    if net.nNodes-len(pruneIdx)>classes:
                        print('-{}nodes'.format(len(pruneIdx)))
                        net.nodePruning(pruneIdx,pruneIdx.size(0))
                        if epoch<0:
                            net = pickle.loads(pickle.dumps(net))
                        net = copy.deepcopy(net)

                        # change si buffer
                        if si:
                            cnt = 0
                            for idx in pruneIdx:
                                pIdx = idx-cnt
                                if runs>0:
                                    p_old['linear__weight']       = (deleteRowTensor(p_old['linear__weight'], pIdx)).detach()
                                    p_old['linear__bias']         = (deleteRowTensor(p_old['linear__bias'], pIdx)).detach()
                                    p_old['linearOutput__weight'] = (deleteColTensor(p_old['linearOutput__weight'], pIdx)).detach()
                                    W['linear__weight']           = (deleteRowTensor(W['linear__weight'], pIdx)).detach()
                                    W['linear__bias']             = (deleteRowTensor(W['linear__bias'], pIdx)).detach()
                                    W['linearOutput__weight']     = (deleteColTensor(W['linearOutput__weight'], pIdx)).detach()
                                for n, p in net.network.named_buffers():
                                    if 'si_init' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteColTensor(p, pIdx)).detach())
                                    elif 'si_optim' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteColTensor(p, pIdx)).detach())
                                    elif 'si_omega' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteColTensor(p, pIdx)).detach())
                                cnt = cnt + 1


        # feedforward
        net.network.train()
        net.network = net.network.to(device)

        optimizer = torch.optim.SGD(net.network.conv1d.parameters(), lr=lrGen, momentum=0.95)
        optimizer.add_param_group({'params': net.network.linear.parameters()})
        optimizer.add_param_group({'params': net.network.biasDecoder})
        optimizer.add_param_group({'params': net.network.sensDecoder.parameters()})

        h, _ = net.network(x=data1, mode=1)
        _, rSensor = net.network(h=h, mode=2)

        loss = criterionGen(rSensor.unsqueeze(0), data1)
        # backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if self_label:
            if epoch==-1:
                for k,idx in enumerate(indices):
                    if idx>(batchSize-1):
                        error = torch.mean(torch.abs(rSensor[k]-data1[k])).detach().clone()
                        pseu_er.append(error)
                        pseu_idx.append(idx-batchSize)

    net.network.er_un = pseu_er
    net.network.rc_idx = pseu_idx

    return net

def DisTrain(net, input1, targets, spc, criterion, lr, acm, ps_idx, runs, epoch, device):
    print('discriminative training...')
    # flags
    grownode = False
    prunenode = False

    # shuffle the data
    nData = input1.shape[0]
    shuffled_indices = torch.randperm(nData)

    input1 = input1.to(device)
    targets = targets.to(device)
    targets_spc = F.one_hot(targets).float()
    targets_spc = targets_spc.to(device)

    if ps_idx is not None:
        net.network.er_min = min(net.network.er_un)
        net.network.er_max = max(net.network.er_un)

    for i in range(nData):
        # load data
        indices = shuffled_indices[i:(i + 1)]

        data1 = input1[indices]
        target_ = targets[indices]
        target_spc = targets_spc[indices]

        # grow/prune
        if flexible:
            grow, prune = spc.update_dis(net, acm, target_spc, epoch)

            if grow and spc.count >= 1:
                # grow nodes
                print('{}nodes+{}'.format(net.nNodes, acm.k))
                net.nodeGrowing(nNewNode=acm.k, device=device)
                if epoch < 0:
                    net = pickle.loads(pickle.dumps(net))
                net = copy.deepcopy(net)

                # change si buffer
                if si:
                    newnodes = acm.k
                    newrow = torch.zeros(newnodes, 20).to(device)
                    newcol = torch.zeros(3, newnodes).to(device)
                    p_old['linear__weight'] = (torch.cat((p_old['linear__weight'], newrow), 0)).detach()
                    p_old['linear__bias'] = (torch.cat((p_old['linear__bias'], torch.zeros(newnodes).to(device)), 0)).detach()
                    p_old['linearOutput__weight'] = (torch.cat((p_old['linearOutput__weight'], newcol), 1)).detach()
                    W['linear__weight'] = (torch.cat((W['linear__weight'], newrow), 0)).detach()
                    W['linear__bias'] = (torch.cat((W['linear__bias'], torch.zeros(newnodes).to(device)), 0)).detach()
                    W['linearOutput__weight'] = (torch.cat((W['linearOutput__weight'], newcol), 1)).detach()
                    for n, p in net.network.named_buffers():
                        if 'si_init' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newrow), 0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, torch.zeros(newnodes).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newcol), 1)).detach())
                        elif 'si_optim' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newrow), 0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, torch.zeros(newnodes).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newcol), 1)).detach())
                        elif 'si_omega' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newrow), 0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, torch.zeros(newnodes).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newcol), 1)).detach())

            if prune and not grow and spc.count >= 20:
                # prune node
                Es = sum([acm.wm[i] * net.network.activation(net.network.linear(acm.center[i])) for i in range(acm.k)])
                pruneIdx = findLeastSignificantNode(Es)
                if pruneIdx is not None:
                    if net.nNodes-len(pruneIdx)>classes:
                        print('-nodes from {} to {}'.format(net.nNodes,net.nNodes-len(pruneIdx)))
                        net.nodePruning(pruneIdx, pruneIdx.size(0))
                        if epoch<0:
                            net = pickle.loads(pickle.dumps(net))
                        net = copy.deepcopy(net)

                        # change si buffer
                        if si:
                            cnt = 0
                            for idx in pruneIdx:
                                pIdx = idx-cnt
                                p_old['linear__weight'] = (deleteRowTensor(p_old['linear__weight'], pIdx)).detach()
                                p_old['linear__bias'] = (deleteRowTensor(p_old['linear__bias'], pIdx)).detach()
                                p_old['linearOutput__weight'] = (deleteColTensor(p_old['linearOutput__weight'], pIdx)).detach()
                                W['linear__weight'] = (deleteRowTensor(W['linear__weight'], pIdx)).detach()
                                W['linear__bias'] = (deleteRowTensor(W['linear__bias'], pIdx)).detach()
                                W['linearOutput__weight'] = (deleteColTensor(W['linearOutput__weight'], pIdx)).detach()
                                for n, p in net.network.named_buffers():
                                    if 'si_init' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteColTensor(p, pIdx)).detach())
                                    elif 'si_optim' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteColTensor(p, pIdx)).detach())
                                    elif 'si_omega' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),(deleteColTensor(p, pIdx)).detach())
                                cnt = cnt + 1

        # feedforward
        net.network.train()
        net.network = net.network.to(device)

        optimizer = torch.optim.SGD(net.network.conv1d.parameters(), lr=lr, momentum=0.95)
        optimizer.add_param_group({'params': net.network.linear.parameters()})
        optimizer.add_param_group({'params': net.network.linearOutput.parameters()})

        ### auto regularisation
        if si:
            thresval = (2*batchSize-1) if augment else (batchSize-1)
            if torch.max(indices) > thresval or i == (nData-1):
                for n, p in net.network.named_parameters():
                    if 'Decoder' not in n:
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            # Find/calculate new values for quadratic penalty on parameters
                            p_init = getattr(net.network, '{}_si_init'.format(n))
                            p_current = p.detach().clone()
                            p_change = p_current - p_init
                            omega_add = W[n]/(p_change ** 2 + epsilon)
                            try:
                                omega = getattr(net.network, '{}_si_omega'.format(n))
                            except AttributeError:
                                omega = p.detach().clone().zero_()
                            omega_new = omega + omega_add

                            # print(torch.max(omega_new))
                            if torch.max(omega_new)>100:
                                omega_new = omega_new/torch.norm(omega_new)

                            # Store these new values in the model
                            net.network.register_buffer('{}_si_omega'.format(n), omega_new)
                            net.network.register_buffer('{}_si_init'.format(n), p.data.clone())
                            W[n] = p.data.clone().zero_()

        scores = net.network(x=data1, mode=3)
        loss = criterion(scores, target_)

        if si:
            thresval = (2 * batchSize - 1) if augment else (batchSize - 1)
            if torch.max(indices) > thresval:
                losses = []
                for n, p in net.network.named_parameters():
                    if 'Decoder' not in n:
                        if p.requires_grad:
                            # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                            n = n.replace('.', '__')
                            optim_values = getattr(net.network, '{}_si_optim'.format(n))
                            wImportance = getattr(net.network, '{}_si_omega'.format(n))
                            # Calculate SI's surrogate loss, sum over all parameters
                            # p_current = p.detach().clone()
                            a = 2 if augment else 1
                            er_r = net.network.er_un[(torch.stack(net.network.rc_idx) == (torch.max(indices)-a*batchSize)).nonzero()]
                            alpha = (er_r - net.network.er_min) / (net.network.er_max - net.network.er_min)
                            if alpha<0:
                                print(alpha)
                            losses.append((alpha * wImportance * ((p - optim_values) ** 2)).sum())
                reg = sum(losses)
                loss.add_(reg, alpha=0.5)
                # loss = loss+0.5*reg

        # backward
        optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)
        optimizer.step()

        if si:
            for n, p in net.network.named_parameters():
                if 'Decoder' not in n:
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            thresval0 = 2 * batchSize if augment else batchSize
                            if torch.max(indices) < thresval0:
                                W[n] = W[n] + (-p.grad * (p.detach() - p_old[n]))
                                net.network.register_buffer('{}_si_optim'.format(n), p.data.clone())

    return net


class SPC_G():
    def __init__(self):
        self.bias = []
        self.miu_bias_t = 0
        self.std_bias_t = 0.001
        self.miu_bias_min = 100
        self.std_bias_min = 100
        self.var = []
        self.miu_var_t = 0
        self.std_var_t = 0.001
        self.miu_var_min = 100
        self.std_var_min = 100
        self.count = 0
        self.grow = False
        self.prune = False

    def update_gen(self, net, acm, x1, epoch):
        if epoch > 0:
            self.grow = False
            self.prune = False
        else:
            with torch.no_grad():
                net.network.eval()
                self.count = self.count + 1
                Eh = [acm.wm[i] * net.network.activation(net.network.linear(acm.center[i])) for i in range(acm.k)]
                Eh = sum(Eh)
                Ex = net.network.activation(F.linear(Eh, net.network.linear.weight.t()) + net.network.biasDecoder)
                Ex = Ex.unsqueeze(0)
                rsen = net.network.sensDecoder(Ex)
                Eh2 = [acm.wm[i] * net.network.activation(net.network.linear(acm.center[i] ** 2)) for i in range(acm.k)]
                Eh2 = sum(Eh2)
                Ex2 = net.network.activation(F.linear(Eh2, net.network.linear.weight.t()) + net.network.biasDecoder)
                Ex2 = Ex2.unsqueeze(0)
                rsen2 = net.network.sensDecoder(Ex2)

                bias = torch.mean(torch.abs(rsen - x1)**2).item()
                self.bias.append(bias)
                var = torch.mean(torch.abs(rsen2 - x1 ** 2)).item()
                self.var.append(var)

                self.miu_bias_t = np.mean(self.bias)
                self.std_bias_t = np.std(self.bias)
                self.miu_var_t = np.mean(self.var)
                self.std_var_t = np.std(self.var)

                # node growing
                if self.count <= 1 or self.grow:
                    self.miu_bias_min = copy.deepcopy(self.miu_bias_t)
                    self.std_bias_min = copy.deepcopy(self.std_bias_t)
                else:
                    if self.miu_bias_t < self.miu_bias_min:
                        self.miu_bias_min = copy.deepcopy(self.miu_bias_t)
                    if self.std_bias_t < self.std_bias_min:
                        self.std_bias_min = copy.deepcopy(self.std_bias_t)

                equ1 = self.miu_bias_t + self.std_bias_t
                equ2 = self.miu_bias_min + (1.2 * np.exp(-bias) + 0.8) * self.std_bias_min

                if equ1 >= equ2 and self.count > 10:
                    self.grow = True
                else:
                    self.grow = False

                # prune node
                if self.count <= 10 or self.prune:
                    self.miu_var_min = copy.deepcopy(self.miu_var_t)
                    self.std_var_min = copy.deepcopy(self.std_var_t)
                else:
                    if self.miu_var_t < self.miu_var_min:
                        self.miu_var_min = copy.deepcopy(self.miu_var_t)
                    if self.std_var_t < self.std_var_min:
                        self.std_var_min = copy.deepcopy(self.std_var_t)

                equ3 = self.miu_var_t + self.std_var_t
                equ4 = self.miu_var_min + 2 * (1.2 * np.exp(-var) + 0.8) * self.std_var_min

                if equ3 >= equ4 and not self.grow and self.count >= 20:
                    self.prune = True
                else:
                    self.prune = False

        return self.grow, self.prune

class SPC_D():
    def __init__(self):
        self.bias = []
        self.miu_bias_t = 0
        self.std_bias_t = 0.001
        self.miu_bias_min = 100
        self.std_bias_min = 100
        self.var = []
        self.miu_var_t = 0
        self.std_var_t = 0.001
        self.miu_var_min = 100
        self.std_var_min = 100
        self.count = 0
        self.grow = False
        self.prune = False

    def update_dis(self, net, acm, label, epoch):
        if epoch > 0:
            self.grow = False
            self.prune = False
        else:
            with torch.no_grad():
                net.network.eval()
                self.count = self.count + 1
                Es = [acm.wm[i] * net.network.activation(net.network.linear(acm.center[i])) for i in range(acm.k)]
                Es = sum(Es)
                Ey = net.network.linearOutput(Es)
                Ey = F.softmax(Ey, dim=0)
                Es2 = [acm.wm[i] * net.network.activation(net.network.linear(acm.center[i]) ** 2) for i in range(acm.k)]
                Es2 = sum(Es2)
                Ey2 = net.network.linearOutput(Es2)
                Ey2 = F.softmax(Ey2, dim=0)

                bias = torch.norm((Ey - label) ** 2).item()
                self.bias.append(bias)
                var  = torch.norm(Ey2 - label ** 2).item()
                self.var.append(var)

                self.miu_bias_t = np.mean(self.bias)
                self.std_bias_t = np.std(self.bias)
                self.miu_var_t = np.mean(self.var)
                self.std_var_t = np.std(self.var)

                # node growing
                if self.count <= 1 or self.grow:
                    self.miu_bias_min = copy.deepcopy(self.miu_bias_t)
                    self.std_bias_min = copy.deepcopy(self.std_bias_t)
                else:
                    if self.miu_bias_t < self.miu_bias_min:
                        self.miu_bias_min = copy.deepcopy(self.miu_bias_t)
                    if self.std_bias_t < self.std_bias_min:
                        self.std_bias_min = copy.deepcopy(self.std_bias_t)

                equ1 = self.miu_bias_t + self.std_bias_t
                equ2 = self.miu_bias_min + (1.2 * np.exp(-bias) + 0.8) * self.std_bias_min

                if equ1 >= equ2 and self.count > 10:
                    self.grow = True
                else:
                    self.grow = False

                # prune node
                if self.count <= 10 or self.prune:
                    self.miu_var_min = copy.deepcopy(self.miu_var_t)
                    self.std_var_min = copy.deepcopy(self.std_var_t)
                else:
                    if self.miu_var_t < self.miu_var_min:
                        self.miu_var_min = copy.deepcopy(self.miu_var_t)
                    if self.std_var_t < self.std_var_min:
                        self.std_var_min = copy.deepcopy(self.std_var_t)

                equ3 = self.miu_var_t + self.std_var_t
                equ4 = self.miu_var_min + 2 * (1.2 * np.exp(-var) + 0.8) * self.std_var_min

                if equ3 >= equ4 and not self.grow and self.count >= 20:
                    self.prune = True
                else:
                    self.prune = False

        return self.grow, self.prune

def plot(a,b,picname):
    x = np.arange(len(a[0]))
    plt.plot(x, a[0], color="r", linestyle="-", marker="^", label=b[0], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[1], color="b", linestyle="-", marker="s", label=b[1], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[2], color="g", linestyle="-", marker=",", label=b[2], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[3], color="m", linestyle="-", marker=".", label=b[3], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[4], color="y", linestyle="-", marker="x", label=b[4], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[5], color="c", linestyle="-", marker="o", label=b[5], linewidth=1)
    plt.legend(loc='lower right')
    plt.xlabel('data batch')
    plt.ylabel('accuracy')
    plt.title('1D CNN')
    plt.savefig(picname,dpi=200)
    plt.show()


def loadfile(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def plot_metrics(a,b,key):
    x = np.arange(len(a[0]))
    plt.plot(x, a[0], color="r", linestyle="-", marker="^", label=b[0], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[1], color="b", linestyle="-", marker="s", label=b[1], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[2], color="g", linestyle="-", marker=",", label=b[2], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[3], color="m", linestyle="-", marker=".", label=b[3], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[4], color="y", linestyle="-", marker="x", label=b[4], linewidth=1)
    plt.legend(loc='lower right')
    plt.plot(x, a[5], color="c", linestyle="-", marker="o", label=b[5], linewidth=1)
    plt.legend(loc='lower right')
    plt.xlabel('data batch')
    plt.ylabel(key)
    plt.title('1D CNN')
    plt.show()


def loadresult(file,pic,metric):
    ave = [[]]
    for i in range(len(b)-1):
        ave.append([])
    # get performance dict
    f=open(file,'rb')
    r=pickle.load(f)
    f.close()
    for i in range(len(b)):
#         print(b[i])
#         # print('si:{} selflabel:{} flexibel:{}'.format(r['SI'][i],r['Selflabel'][i],r['Flexiblelayer'][i]))
# #         print(r['Performance'][i])
#         print(np.mean(r['Performance'][i][1:]))
        ave[i].append(np.mean(r['Performance'][i][1:]))
    # print('nodes:',r['Nodes'])
    result = []
    for a in ave:
        result.append(a)
    print('acc:',result)
    [overallacc[i].append(result[i][0]) for i in range(len(b))]
    nodes = r['Nodes']
    [overallnode[i].append(nodes[i]) for i in range(len(b))]
    ### Read Images
    if os.path.isfile(pic):
        img = mpimg.imread(pic)
        plt.imshow(img)

    # get metrics dict
    m = loadfile(metric)
    performs = [[],[],[]]
    for i,key in enumerate(['F1','precision','recall']):
        for k in m:
            f = []
            for l in range(len(k)):
                f.append(k[l][key])
            performs[i].append(f)
        # plot_metrics(performs[i],b,key)
    f1result = []
    precision = []
    recall = []
    for a in range(len(b)):
        f1result.append(np.mean(performs[0][a][1:]))
        precision.append(np.mean(performs[1][a][1:]))
        recall.append(np.mean(performs[2][a][1:]))
    print('F1:',f1result)
    print('precision:', precision)
    print('recall:', recall)
    print('nodes:', nodes)
    [overallf1[i].append(f1result[i]) for i in range(len(b))]
    [overallprec[i].append(precision[i]) for i in range(len(b))]
    [overallrecall[i].append(recall[i]) for i in range(len(b))]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

########################################################################################
############################## run ablation study ######################################
########################################################################################
### variables
SI         =  [False, False, True , False, True , True]
Self_label =  [False, True , True , False, True , True]
Flexible   =  [False, True , False, True , True , True]
Clustering =  [False, True , True , True , True , True]
Augment    =  [False, True , True , True , False, True]

# model A : all off, model B: evolving off, model C: pseudo&si off
b = ['all off', 'si off', 'evolving off', 'pseudo&si off','augment off','parsnet++']

########################################################################################
###############################     ParsNet++     ######################################
########################################################################################
# ### variables
# SI         =  [True]
# Self_label =  [True]
# Flexible   =  [True]
# Clustering =  [True]
# Augment    =  [True]
# b = ['parsnet++']

epochs = 10
Epoch = 5
init_batch = 5              # batches for initialization of parsnet+ network
labeled_proportion = 0.5
batchSize = 50
minibatchsize = 5           # if minibatchsize=1, node growing&pruning criterion needs to be changed
noise_std = 0.001
epsilon = 0.001

### network parameters
nInputSensor = 48
classes = 3
lr = 0.01
lrGen = 0.01

### parameter stamp
paramstamp = 'init_batch{}_epochs{}_labeled{}_batchsize{}_epsilon{}_noisestd{}_lrDis{}_lrGen{}_0.6_0.8_0.8'.format(
    init_batch,epochs,labeled_proportion,batchSize,epsilon,noise_std,lr,lrGen)

### select random seed
seedlist = random.sample(range(0, 100), Epoch)
# ### To access the result in paper, use seeds below
# seedlist = [5,13,57,16,33]
print('seed:',seedlist)

overallacc = [[],[],[],[],[],[]]
overallf1 = [[],[],[],[],[],[]]
overallprec = [[],[],[],[],[],[]]
overallrecall = [[],[],[],[],[],[]]
overallnode = [[],[],[],[],[],[]]
for e in range(len(seedlist)):
    randomseed = seedlist[e]
    print('\nROUND {} >>>>>>>>>>>>>>>>>>>'.format(e))
    fileName = 'results_cur/Sensor_accuracy_{}-seed{}.txt'.format(paramstamp,randomseed)
    picName = 'figures_cur/Sensor_pic_{}-seed{}.jpg'.format(paramstamp,randomseed)
    metricfile = 'results_cur/Sensor_metrics_{}-seed{}.txt'.format(paramstamp,randomseed)
    
    # creat result folder if not exists
    if not os.path.isdir('results_cur'):
        os.mkdir('results_cur')
    if not os.path.isdir('figures_cur'):
        os.mkdir('figures_cur')
        
    if not os.path.isfile(fileName):
        print(" ...running: ... ")
        Metrics = []
        Accuracy = []
        depth = []
        for exp in range(len(SI)):

            # set random seed
            setup_seed(randomseed)

            ### load dataset
            print('...loading data...')
            labeled_sensorloader, unlabeled_sensorloader = dataprep_weakly.load_sensor(labeled_proportion, batchSize)
            nclass = classes

            criterion = nn.CrossEntropyLoss()
            criterionGen = nn.MSELoss()             # Generative learning criterion
            net = smallPars(nInputSensor,classes,classes)

            si = SI[exp]
            self_label = Self_label[exp]
            flexible = Flexible[exp]
            Acm = Clustering[exp]
            augment = Augment[exp]


            # accuracy matrix for each training
            acc = []
            metric = []
            for i, x in enumerate(zip(labeled_sensorloader, unlabeled_sensorloader)):
                print(i, '-th batch...')

                sen = x[0][0]
                u_sen = x[1][0]
                target = x[0][1].int().long()
                untarget = x[1][1].int().long()

                sen = sen.to(device)
                u_sen = u_sen.to(device)
                target = target.to(device)
                untarget = untarget.to(device)

                # initialise clustering
                if i == 0:
                    net.network.eval()
                    net.network = net.network.to(device)
                    with torch.no_grad():
                        _, cat = net.network(x=sen, mode=1)
                        acm = ACM(cat[0], target[0], classes)
                        acm.batchSize = batchSize
                        spcD = SPC_D()
                        spcG = SPC_G()

                # testing
                test_sen = torch.cat((sen, u_sen), 0)
                test_label = torch.cat((target, untarget), 0)
                scores, loss, accuracy, test_time, fpr = pre_test(net, test_sen, test_label, criterion, device)
                acc.append(accuracy)
                metric.append(fpr)

                # data augmentation
                if augment:
                    noisysen = addGaussianNoise1D(sen,std=noise_std, device=device)

                tic = time.time()

                if i < init_batch:
                    for epoch in range(epochs):
                        print('epoch{}'.format(epoch))
                        if epoch == 0:
                            # clustering
                            if Acm:
                                acm = clustering(net, acm, sen, target, epoch, device)
                        # Generative training
                        G_sens = torch.cat((sen, u_sen), 0)
                        net = GenTrain(net, G_sens, spcG, criterionGen, lrGen, acm, i, epoch, device)
                        # plt.plot(np.arange(len(spcG.equ1)),spcG.equ1,color='r')
                        # plt.plot(np.arange(len(spcG.equ2)), spcG.equ2,color='b')
                        # plt.show()

                        # auto regularisation
                        if si and self_label:
                            W = {}
                            p_old = {}
                            for n, p in net.network.named_parameters():
                                if 'Decoder' not in n:
                                    n = n.replace('.', '__')
                                    net.network.register_buffer('{}_si_init'.format(n), p.data.clone())
                                    W[n] = p.data.clone().zero_()
                                    p_old[n] = p.data.clone()

                        # Discriminative training
                        D_sens_ = sen
                        targets_ = target
                        if augment:
                            D_sens_ = torch.cat((sen, noisysen), 0)
                            targets_ = torch.cat((target, target), 0)
                        net = DisTrain(net, D_sens_, targets_, spcD, criterion, lr, acm, None, i, epoch, device=device)

                else:
                    epoch = -1
                    # clustering
                    if Acm:
                        acm = clustering(net, acm, sen, target, epoch, device)

                    # pseudo samples
                    if self_label:
                        pseu_data, pseu_sen, pseu_label, p_idx = selflabel(net,acm,u_sen)

                    '''# update cluster center
                    if len(pseu_label)!=0:
                        if len(pseu_label)==1:
                            pseu_data = pseu_data[0]
                            pseu_label = pseu_label[0]
                            acm.pseudo_update(pseu_data,pseu_label)
                        else:
                            pseu_data = (torch.cat(pseu_data,dim=0)).view(len(pseu_label),532)
                            pseu_label = torch.tensor(pseu_label)
                            acm.pseudo_update(pseu_data,pseu_label)'''

                    # Generative training
                    G_sens = torch.cat((sen, u_sen), 0)
                    net = GenTrain(net, G_sens, spcG, criterionGen, lrGen, acm, i, epoch, device)

                    # auto regularisation
                    if si and self_label:
                        W = {}
                        p_old = {}
                        for n, p in net.network.named_parameters():
                            if 'Decoder' not in n:
                                n = n.replace('.', '__')
                                net.network.register_buffer('{}_si_init'.format(n), p.data.clone())
                                W[n] = p.data.clone().zero_()
                                p_old[n] = p.data.clone()

                    # Discriminative training
                    D_sens_ = sen
                    targets_ = target
                    if augment:
                        D_sens_ = torch.cat((sen, noisysen), 0)
                        targets_ = torch.cat((target, target), 0)
                    if self_label:
                        if len(pseu_label) > 0:
                            ### data processing
                            pseu_sen = torch.stack(pseu_sen)
                            pseu_label = torch.stack(pseu_label, dim=0).long()

                            D_sens = torch.cat((D_sens_, pseu_sen), 0)
                            targets = torch.cat((targets_, pseu_label), 0)

                            # ### calculate reconstruction error
                            # error_pseudo = []
                            # for id in p_idx:
                            #     rc_r = net.network.er_un[(torch.stack(net.network.rc_idx) == id).nonzero()]
                            #     error_pseudo.append(rc_r)
                            # net.network.er_min = min(error_pseudo)
                            # net.network.er_max = max(error_pseudo)

                            net = DisTrain(net, D_sens, targets, spcD, criterion, lr, acm, p_idx, i, epoch, device=device)
                        else:
                            D_sens = D_sens_
                            targets = targets_
                            net = DisTrain(net, D_sens, targets, spcD, criterion, lr, acm, None, i, epoch, device=device)
                    else:
                        D_sens = D_sens_
                        targets = targets_
                        net = DisTrain(net, D_sens, targets, spcD, criterion, lr, acm, None, i, epoch, device=device)

                toc = time.time()
                training_time = toc - tic
                print('Training time:', training_time)
            print(acc)
            Accuracy.append(acc)
            Metrics.append(metric)
            depth.append(net.nNodes)

            print(paramstamp)
            print("pseudo label:{}, si:{}, flexible:{}, augment:{}, acm:{}".format(self_label,
                                                                                   si, flexible, augment, Acm))
            print('average accuracy:', np.mean(acc))
            print('network node no is',net.nNodes)

        # save results to file
        output_file = open(fileName, 'wb')
        results = {'SI': SI, 'Selflabel': Self_label, 'Nodes': depth,
                   'Flexiblelayer': Flexible, 'ACM': Clustering, 'NoisyData': Augment, 'Performance': Accuracy}
        pickle.dump(results, output_file)
        output_file.close()
        # save results to file
        metric_file = open(metricfile, 'wb')
        pickle.dump(Metrics, metric_file)
        metric_file.close()
        # # generate image
        # plot(Accuracy,b,picName)

        if len(b) != len(Accuracy):
            raise Exception('size error')

        print('...........results...........')
        print(len(Accuracy[0]))
        print('Average')
        for head, re in zip(b, Accuracy):
            print(head)
            print(np.mean(re))
        print('F1')
        for i in range(len(SI)):
            print(b[i])
            f1 = []
            for f in Metrics[i]:
                f1.append(f['F1'])
            print(np.mean(f1))

        print('...........results [1:]...........')
        print('Average')
        for head,re in zip(b,Accuracy):
            print(head)
            print(np.mean(re[1:]))
        print('F1')
        for i in range(len(SI)):
            print(b[i])
            f1 = []
            for f in Metrics[i]:
                f1.append(f['F1'])
            print(np.mean(f1[1:]))
        print(depth)

    else:
        loadresult(fileName,picName,metricfile)
        if e == (len(seedlist)-1):
            print('------------------')
            print('average accuracy:')
            [print('{:.4f} +/- {:.4f}'.format(np.mean(n),np.std(n))) for n in overallacc]
            print('f1')
            [print('{:.4f} +/- {:.4f}'.format(np.mean(n), np.std(n))) for n in overallf1]
            print('precision')
            [print('{:.4f} +/- {:.4f}'.format(np.mean(n), np.std(n))) for n in overallprec]
            print('recall')
            [print('{:.4f} +/- {:.4f}'.format(np.mean(n), np.std(n))) for n in overallrecall]
            print('nodes')
            [print('{:.1f} +/- {:.1f}'.format(np.mean(n),np.std(n))) for n in overallnode]

