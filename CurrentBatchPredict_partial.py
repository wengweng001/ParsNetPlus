import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F
import dataprep_weakly
import time
import numpy as np
from numpy import linalg as LA
import copy
import pickle
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score
import matplotlib.image as mpimg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class sensorNet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(sensorNet, self).__init__()
        # cnn layer
        self.cnn1d_1 = nn.Conv1d(in_channels = in_chan, out_channels = out_chan, kernel_size = 1, stride = 1, bias=True)
        #         self.cnn1d_2 = nn.Conv1d(in_channels = out_chan, out_channels = 20, kernel_size = 1, stride = 1, bias=True)
        self.cnn1d_2 = nn.Conv1d(in_channels = out_chan, out_channels = 40, kernel_size = 1, stride = 1, bias=True)
        self.cnn1d_3 = nn.Conv1d(in_channels = 40, out_channels = 20, kernel_size = 1, stride = 1, bias=True)

        nn.init.xavier_uniform_(self.cnn1d_1.weight)
        self.cnn1d_1.bias.data.zero_()
        nn.init.xavier_uniform_(self.cnn1d_2.weight)
        self.cnn1d_2.bias.data.zero_()
        nn.init.xavier_uniform_(self.cnn1d_3.weight)
        self.cnn1d_3.bias.data.zero_()

        self.activation = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout()

    def forward(self, x):
        x = self.cnn1d_1(x)
        x = self.activation(x)
        #         x = self.dropout(x)
        x = self.cnn1d_2(x)
        x = self.activation(x)
        #         x = self.dropout(x)
        x = self.cnn1d_3(x)
        x = self.activation(x)
        #         x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x


class DecoderS(nn.Module):
    def __init__(self, sensors, h_dim=20):
        super(DecoderS, self).__init__()
        self.tran_1 = nn.ConvTranspose1d(in_channels=h_dim, out_channels=40, kernel_size=1, stride=1, bias=True)
        self.tran_2 = nn.ConvTranspose1d(in_channels=40, out_channels=60, kernel_size=1, stride=1, bias=True)
        self.tran_3 = nn.ConvTranspose1d(in_channels=60, out_channels=sensors, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.tran_1(x)
        x = self.relu(x)
        x = self.tran_2(x)
        x = self.relu(x)
        x = self.tran_3(x)
        x = self.sigm(x)
        x = x.squeeze()
        return x


def addGaussianNoise1D(x,std = 0.001):
    # noiseIntensity: sigma in gaussian noise function
    nData, nFeat = x.shape
    y = x.clone().detach()
    for i,_ in enumerate(x):
        noise = torch.normal(0, std, size = [nFeat,])
        noise = noise.to(device)
        y[i] = x[i] + noise
    return y


class ACM():
    def __init__(self, sample, label, nclass):
        super(ACM, self).__init__()
        self.k = 1
        self.classes = nclass
        self.car = [1]
        self.center = [sample]
        self.span = [0]
        self.l2d = [[]]
        self.phi = [[]]
        self.count = 0
        self.wm = []
        self.carom = [[0], [0], [0]]
        self.carom[label][0] = self.carom[label][0] + 1

    def criterion(self, z, target):
        with torch.no_grad():
            for i, data in enumerate(z):
                #             print('data', i, '...')
                self.count = self.count + 1

                d = []
                for k in range(self.k):
                    d.append(torch.norm(data - self.center[k]))
                idx = np.argmin(d)
                dis = min(d)
                k3 = 2 * torch.exp(-dis ** 2) + 1
                cri = np.mean((self.l2d[idx])) + k3 * np.std(self.l2d[idx])  # equ.13 right side

                if dis > cri and self.count > 10 and (i+1) != batchSize:  # 10 data points for initialise
                    #             if dis > cri and i!=99:
                    grow_cluster = 1
                    self.k = self.k + 1
                    self.car.append(1)
                    self.center.append(data)
                    self.l2d.append([])

                    self.span.append(0)
                    self.span = [j + 1 for j in self.span]
                    self.phi.append([torch.exp(-(torch.norm(data - self.center[-1])) ** 2).item()])
                    [self.carom[o].append(0) for o in range(3)]
                    self.carom[target[i]][self.k - 1] = self.carom[target[i]][self.k - 1] + 1
                else:
                    grow_cluster = 0

                    self.span = [j + 1 for j in self.span]
                    #                 self.phi[idx].append(torch.exp(-(torch.norm(data - self.center[idx])) ** 2))

                    self.center[idx] = self.center[idx] + (data - self.center[idx]) / (self.car[idx] + 1)  # equ.14
                    self.car[idx] = self.car[idx] + 1
                    self.l2d[idx].append(dis.item())

                    self.phi[idx].append(torch.exp(-(torch.norm(data - self.center[idx])) ** 2).item())
                    self.carom[target[i]][idx] = self.carom[target[i]][idx] + 1

                act = [sum(self.phi[i]) / self.span[i] for i in range(self.k)]  # equ.15

                if (i+1) % batchSize == 0 and self.k > self.classes and grow_cluster == 0:
                    prune = 0
                    print('cluster no.:', self.k)
                    for pruneidx, val in enumerate(act):
                        if val <= abs(np.mean(act) - 0.5 * np.std(act)):  # equ.16
                            pidx = pruneidx - prune
                            self.k = self.k - 1
                            del self.car[pidx]
                            del self.center[pidx]
                            del self.span[pidx]
                            del self.l2d[pidx]
                            del self.phi[pidx]
                            del self.carom[0][pidx]
                            del self.carom[1][pidx]
                            del self.carom[2][pidx]
                            prune = prune + 1
                    print('prune clusters:', prune)
                    print('cluters*:', self.k)

    def pseudo_criterion(self, z, target):
        with torch.no_grad():
            num = sum([(self.carom[target][i]/sum([self.carom[o][i]for o in range(3)]))*(self.car[i]/sum(self.car))*np.exp(-(np.linalg.norm((z - self.center[i]).cpu(),2) ** 2))
                       for i in range(self.k)])
            den_ = [sum([(self.carom[classes][i] / sum([self.carom[o][i] for o in range(3)])) * (
                        self.car[i] / sum(self.car)) * np.exp(-(np.linalg.norm((z - self.center[i]).cpu(),2)**2)) for i in
                   range(self.k)]) for classes in range(3)]
            den = sum(den_)
        return num/den


def clustering(net, acm, data1, label, epoch, device):
    if epoch==0 or epoch==-1:
        data1 = data1.to(device)
        label = label.to(device)

        net.network.eval()
        net.network = net.network.to(device)
        with torch.no_grad():
            _, h = net.network(x=data1,mode=1)
        h = h.to(device)
        acm.criterion(h, label)
        print('clusters:{}'.format(acm.k))
        denom = 0
        for m in range(acm.k):
            denom = denom + acm.phi[m][-1] * acm.car[m]
        acm.wm = [acm.phi[i][-1] * acm.car[i] / denom for i in range(acm.k)]
        return acm


class basicPars(nn.Module):
    def __init__(self, nInputSensor, no_hidden, no_output):
        super(basicPars, self).__init__()
        # encoder
        self.conv1d = sensorNet(nInputSensor,60)
        self.linear = nn.Linear(20,no_hidden,bias=True)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, no_output, bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()

        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(20))
        self.sensDecoder = DecoderS(nInputSensor)

        # network para for regularization
        self.weightinit = []
        self.weightoptim = []
        self.weightlt = []
        self.biasinit = []
        self.biasoptim = []
        self.biaslt = []
        self.step = 0
        self.grads = []
        self.er_un = []


    def forward(self,x=None,h=None,mode=1):
        # mode 1. encoder, 2. decoder, 3. predictor
        if mode == 1:
            # encoder
            x = x.unsqueeze(2)
            x_ = self.conv1d(x)
            x = self.linear(x_)
            x = self.activation(x)
            return x,x_

        if mode == 2:
            # decoder
            x = F.linear(h, self.linear.weight.t()) + self.biasDecoder
            x_ = self.activation(x)  # reconstructed x_ in mode:1
            r_sen = self.sensDecoder(x_)
            return x_,r_sen

        if mode == 3:
            # predict
            x = x.unsqueeze(2)
            x = self.conv1d(x)
            x = self.linear(x)
            x = self.activation(x)
            x = self.linearOutput(x)
            return x


def deleteRowTensor(x, index):
    x = x[torch.arange(x.size(0)) != index.cpu()]
    return x
def deleteColTensor(x, index):
    x = x.transpose(1, 0)
    x = x[torch.arange(x.size(0)) != index.cpu()]
    x = x.transpose(1, 0)
    return x


class smallPars():
    def __init__(self, no_input, no_hidden, no_output):
        self.network = basicPars(no_input, no_hidden, no_output)
        self.netUpdateProperties()

    def getNetProperties(self):
        print(self.network)
        print('No. of net inputs features:', self.nNetInput)
        print('No. of net nodes :', self.nNodes)
        print('No. of outputs :', self.nOutputs)
        print('No. of net parameters :', self.nParameters)

    def getNetParameters(self):
        print('Input weight: \n', self.network.linear.weight)
        print('Input bias: \n', self.network.linear.bias)
        print('Bias decoder: \n', self.network.biasDecoder)

        print('Output weight: \n', self.network.linearOutput.weight)
        print('Output bias: \n', self.network.linearOutput.bias)

    def netUpdateProperties(self):
        self.nNetInput = self.network.linear.in_features
        self.nNodes = self.network.linear.out_features
        self.nOutputs = self.network.linearOutput.out_features
        self.nParameters = (self.network.linear.in_features * self.network.linear.out_features +
                            len(self.network.linear.bias.data) + len(self.network.biasDecoder) +
                            self.network.linearOutput.in_features * self.network.linearOutput.out_features +
                            len(self.network.linearOutput.bias.data))

    # ============================= evolving =============================
    def nodeGrowing(self, nNewNode=1, device=torch.device('cpu')):
        nNewNodeCurr = self.nNodes + nNewNode

        # grow node
        newWeight = nn.init.xavier_uniform_(torch.empty(nNewNode, self.nNetInput)).to(device)
        self.network.linear.weight.data = (torch.cat((self.network.linear.weight.data,
                                                     newWeight), 0)).clone()  # grow input weights
        self.network.linear.bias.data = (torch.cat((self.network.linear.bias.data,
                                                   torch.zeros(nNewNode).to(device)), 0)).clone()  # grow input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad

        # grow input in the output layer
        newWeightNext = (nn.init.xavier_uniform_(torch.empty(self.nOutputs, nNewNode))).to(device)
        self.network.linearOutput.weight.data = (torch.cat((self.network.linearOutput.weight.data, newWeightNext), 1)).clone()
        del self.network.linearOutput.weight.grad

        self.network.linearOutput.in_features = self.network.linearOutput.in_features + nNewNode

        self.netUpdateProperties()

    def nodePruning(self, pruneList, nPrunedNode=1):
        nNewNodeCurr = self.nNodes - nPrunedNode  # prune a node

        cal = 0
        for pruneIdx in pruneList:
            # prune node for current layer, output
            pruneIdx=pruneIdx-cal
            self.network.linear.weight.data = (deleteRowTensor((self.network.linear.weight.data),
                                                              pruneIdx)).clone()  # prune input weights
            self.network.linear.bias.data = (deleteRowTensor(self.network.linear.bias.data,
                                                            pruneIdx)).clone()  # prune input bias
            self.network.linear.out_features = nNewNodeCurr
            del self.network.linear.weight.grad
            del self.network.linear.bias.grad

            # delete input of the output layer
            self.network.linearOutput.weight.data = (deleteColTensor(self.network.linearOutput.weight.data,
                                                                     pruneIdx)).clone()
            del self.network.linearOutput.weight.grad
            cal = cal + 1
        del cal

        # update input features
        self.network.linearOutput.in_features -= nPrunedNode

        self.netUpdateProperties()


def findLeastSignificantNode(nodeSig):
    if nodeSig.size(0)>classes:
        idx = torch.nonzero(nodeSig<=torch.mean(nodeSig)-0.5*torch.std(nodeSig))
        # print(idx)
        return idx


def selflabel(net,acm,u_sen):
    scores = net.network(x=u_sen, mode=3)
    softscores = F.softmax(scores.data,dim=1)
    top2, ranks = torch.topk(softscores,2)
    pseu_data = []
    pseu_ss = []
    pseu_y = []
    idx = []
    _, un_h = net.network(x=u_sen, mode=1)
    for k, x in enumerate(zip(top2,ranks)):
        if x[0][0] > 0.55:                   # equ (2)
            h = un_h[k]
            if x[1][0]>2 or x[1][0]<0:
                continue
            predict_acm = acm.pseudo_criterion(h, x[1][0])
            if predict_acm > 0.55:
                cri3 = x[0][0]/(x[0][0]+x[0][1])
                if cri3>0.55:
                    pseu_data.append(h)
                    pseu_ss.append(u_sen[k])
                    pseu_y.append(x[1][0])
                    idx.append(k)
    return pseu_data,pseu_ss,pseu_y,idx


def pre_test(net, data, label, criterion, device):
    start_test = time.time()

    data = data.to(device)
    label = label.to(device)

    net.network.eval()
    net.network = net.network.to(device)

    with torch.no_grad():
        classes = net.network.linearOutput.weight.shape[0]
        ndata = data.shape[0]
        y = torch.zeros(ndata, classes)

        scores = net.network(x=data, mode=3)
        predicts = scores.argmax(dim=1, keepdim=True)

        correct = predicts.eq(label.view_as(predicts)).sum().item()
        accuracy = correct / ndata
        loss = criterion(scores, label)
        f1 = f1_score(label.cpu(), predicts.cpu(), average='macro')
        prec = precision_score(label.cpu(), predicts.cpu(), average='macro')
        recall = recall_score(label.cpu(), predicts.cpu(), average='macro')
        fpr = {'F1':f1,'precision':prec,'recall':recall}

    end_test = time.time()
    test_time = end_test - start_test
    print('Testing loss: {}'.format(loss))
    print('Testing accuracy: {}%'.format(100 * accuracy))
    # print('Testing time: {}s'.format(test_time))
    print('Precision:{:.6f},Recall:{:.6f},F1:{:.6f}'.format(prec,recall,f1))

    return scores, loss, accuracy, test_time, fpr


def GenTrain(net, input1, spc, criterionGen, lrGen, acm, runs, epoch, device):
    # input: input data type 0-original data, 1-augment data, 2-unlabeldata
    print('generative training...')
    # flags
    grownode = False
    prunenode = False


    # shuffle the data
    nData = input1.shape[0]
    shuffled_indices = torch.randperm(nData)

    input1 = input1.to(device)

    iters = int(nData / minibatchsize)


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

        loss = criterionGen(rSensor, data1)
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

    iters = int(nData / minibatchsize)
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
                    p_old['linear__bias'] = (
                        torch.cat((p_old['linear__bias'], torch.zeros(newnodes).to(device)), 0)).detach()
                    p_old['linearOutput__weight'] = (torch.cat((p_old['linearOutput__weight'], newcol), 1)).detach()
                    W['linear__weight'] = (torch.cat((W['linear__weight'], newrow), 0)).detach()
                    W['linear__bias'] = (torch.cat((W['linear__bias'], torch.zeros(newnodes).to(device)), 0)).detach()
                    W['linearOutput__weight'] = (torch.cat((W['linearOutput__weight'], newcol), 1)).detach()
                    for n, p in net.network.named_buffers():
                        if 'si_init' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newrow), 0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n), (
                                    torch.cat((p, torch.zeros(newnodes).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newcol), 1)).detach())
                        elif 'si_optim' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newrow), 0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n), (
                                    torch.cat((p, torch.zeros(newnodes).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newcol), 1)).detach())
                        elif 'si_omega' in n:
                            if 'linear__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newrow), 0)).detach())
                            if 'linear__bias' in n:
                                net.network.register_buffer('{}'.format(n), (
                                    torch.cat((p, torch.zeros(newnodes).to(device)), 0)).detach())
                            if 'linearOutput__weight' in n:
                                net.network.register_buffer('{}'.format(n), (torch.cat((p, newcol), 1)).detach())

            if prune and not grow and spc.count >= 20:
                # prune node
                Es = sum([acm.wm[i] * net.network.activation(net.network.linear(acm.center[i])) for i in range(acm.k)])
                pruneIdx = findLeastSignificantNode(Es)
                if pruneIdx is not None:
                    if net.nNodes - len(pruneIdx) > classes:
                        print('-nodes from {} to {}'.format(net.nNodes, net.nNodes - len(pruneIdx)))
                        net.nodePruning(pruneIdx, pruneIdx.size(0))
                        if epoch < 0:
                            net = pickle.loads(pickle.dumps(net))
                        net = copy.deepcopy(net)

                        # change si buffer
                        if si:
                            cnt = 0
                            for idx in pruneIdx:
                                pIdx = idx - cnt
                                p_old['linear__weight'] = (deleteRowTensor(p_old['linear__weight'], pIdx)).detach()
                                p_old['linear__bias'] = (deleteRowTensor(p_old['linear__bias'], pIdx)).detach()
                                p_old['linearOutput__weight'] = (
                                    deleteColTensor(p_old['linearOutput__weight'], pIdx)).detach()
                                W['linear__weight'] = (deleteRowTensor(W['linear__weight'], pIdx)).detach()
                                W['linear__bias'] = (deleteRowTensor(W['linear__bias'], pIdx)).detach()
                                W['linearOutput__weight'] = (deleteColTensor(W['linearOutput__weight'], pIdx)).detach()
                                for n, p in net.network.named_buffers():
                                    if 'si_init' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteColTensor(p, pIdx)).detach())
                                    elif 'si_optim' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteColTensor(p, pIdx)).detach())
                                    elif 'si_omega' in n:
                                        if 'linear__weight' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteRowTensor(p, pIdx)).detach())
                                        if 'linear__bias' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteRowTensor(p, pIdx)).detach())
                                        if 'linearOutput__weight' in n:
                                            net.network.register_buffer('{}'.format(n),
                                                                        (deleteColTensor(p, pIdx)).detach())
                                cnt = cnt + 1

        # feedforward
        net.network.train()
        net.network = net.network.to(device)

        optimizer = torch.optim.SGD(net.network.conv1d.parameters(), lr=lr, momentum=0.95)
        optimizer.add_param_group({'params': net.network.linear.parameters()})
        optimizer.add_param_group({'params': net.network.linearOutput.parameters()})

        ### auto regularisation
        if si:
            thresval = (2 * batchSize - 1)
            if torch.max(indices) > thresval or i == (nData - 1):
                for n, p in net.network.named_parameters():
                    if 'Decoder' not in n:
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            # Find/calculate new values for quadratic penalty on parameters
                            p_init = getattr(net.network, '{}_si_init'.format(n))
                            p_current = p.detach().clone()
                            p_change = p_current - p_init
                            omega_add = W[n] / (p_change ** 2 + epsilon)
                            try:
                                omega = getattr(net.network, '{}_si_omega'.format(n))
                            except AttributeError:
                                omega = p.detach().clone().zero_()
                            omega_new = omega + omega_add

                            # print(torch.max(omega_new))
                            if torch.max(omega_new) > 100:
                                omega_new = omega_new / torch.norm(omega_new)

                            # Store these new values in the model
                            net.network.register_buffer('{}_si_omega'.format(n), omega_new)
                            net.network.register_buffer('{}_si_init'.format(n), p.data.clone())
                            W[n] = p.data.clone().zero_()

        scores = net.network(x=data1, mode=3)
        loss = criterion(scores, target_)

        if si:
            thresval = (2 * batchSize - 1)
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
                            a = 2
                            er_r = net.network.er_un[
                                (torch.stack(net.network.rc_idx) == (torch.max(indices) - a * batchSize)).nonzero()]
                            alpha = (er_r - net.network.er_min) / (net.network.er_max - net.network.er_min)
                            if alpha < 0:
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
                            thresval0 = 2 * batchSize
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


# In[17]:


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
    plt.xlabel('data batch')
    plt.ylabel(key)
    plt.title('1D CNN')
    plt.show()


def loadresult(file,pic,metric):
    ave = []
    # get performance dict
    f=open(file,'rb')
    r=pickle.load(f)
    f.close()
    for i in range(len(SI)):
        # print('si:{} selflabel:{} flexibel:{}'.format(r['SI'][i],r['Selflabel'][i],r['Flexiblelayer'][i]))
#         print(r['Performance'][i])
#         print((r['Performance'][i])[1:])
        print(np.mean(r['Performance'][i][1:]))
        meanacc.append(np.mean(r['Performance'][i][1:]))
    ### Read Images
    # if os.path.isfile(pic):
    #     img = mpimg.imread(pic)
    #     plt.imshow(img)

    # get metrics dict
    m = loadfile(metric)
    performs = [[],[],[]]
    for i,key in enumerate(['F1','precision','recall']):
        for k in m:
            f = []
            for l in range(len(k)):
                if l>0:
                    f.append(k[l][key])
            performs[i].append(f)
    print('F1:', np.mean(performs[0]))
    print('precision:', np.mean(performs[1]))
    print('recall:', np.mean(performs[2]))
    met.append(performs)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

### variables
SI =           [True]
Self_label =   [True]
NormOmega =    [True]
Flexible =     [True]
cluster  =     [True]


epochs = 10
Epoch = 5
init_batch = 5              # batches for initialization of parsnet+ network
labeled_proportion = 0.2    # 0.1 0.2 0.3 0.4 0.5
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
# paramstamp = 'init_batch{}_epochs{}_labeled{}_batchsize{}_minibatch{}_epsilon{}_noisestd{}_lrDis{}_lrGen{}'.format(
#     init_batch,epochs,labeled_proportion,batchSize,minibatchsize,epsilon,noise_std,lr,lrGen)
paramstamp = 'init_batch{}_epochs{}_labeled{}_batchsize{}_epsilon{}_noisestd{}_lrDis{}_lrGen{}'.format(
    init_batch,epochs,labeled_proportion,batchSize,epsilon,noise_std,lr,lrGen)
### select random seed
seedlist = random.sample(range(0, 100), Epoch)
# seedlist = [26, 15, 16, 61, 8] #0.1
# seedlist = [26, 15, 16, 61, 8] # 0.2
# seedlist = [42, 70, 41, 38, 61] #0.3
# seedlist = [40, 25, 92, 11, 74] #0.4
# seedlist = [3, 1, 91, 46, 95] #0.5
print('seed:',seedlist)

meanacc = []
met = []
for e in range(Epoch):
    randomseed = seedlist[e]
    print('Seed{}'.format(randomseed))
    fileName = 'results_labelpercent/Sensor_accuracy_{}-seed{}.txt'.format(paramstamp,randomseed)
    picName = 'figures_labelpercent/Sensor_pic_{}-seed{}.jpg'.format(paramstamp,randomseed)
    metricfile = 'results_labelpercent/Sensor_metrics_{}-seed{}.txt'.format(paramstamp,randomseed)
    
    # creat result folder if not exists
    if not os.path.isdir('results_labelpercent'):
        os.mkdir('results_labelpercent')
    # if not os.path.isdir('figures_labelpercent'):
    #     os.mkdir('figures_labelpercent')
        
    if not os.path.isfile(fileName):
        print(" ...running: ... ")
        Metrics = []
        Accuracy = []
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
            normOmega = NormOmega[exp]
            flexible = Flexible[exp]
            Acm = cluster[exp]

            # accuracy matrix for each training
            acc = []
            metric = []
            depth = []
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
                        spcD = SPC_D()
                        spcG = SPC_G()

                # testing
                test_sen = torch.cat((sen, u_sen), 0)
                test_label = torch.cat((target, untarget), 0)
                scores, loss, accuracy, test_time, fpr = pre_test(net, test_sen, test_label, criterion, device)
                acc.append(accuracy)
                metric.append(fpr)

                # data augmentation
                noisysen = addGaussianNoise1D(sen,std=noise_std)

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
                    D_sens_ = torch.cat((sen, noisysen), 0)
                    targets_ = torch.cat((target, target), 0)
                    if self_label:
                        if len(pseu_label) > 0:
                            ### data processing
                            pseu_sen = torch.stack(pseu_sen)
                            pseu_label = torch.stack(pseu_label, dim=0).long()

                            D_sens = torch.cat((D_sens_, pseu_sen), 0)
                            targets = torch.cat((targets_, pseu_label), 0)

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

            print(paramstamp)
            print("pseudo label:{}, si:{}, norm_omega:{}, flexible:{}".format(self_label, si, normOmega, flexible))
            print('average accuracy:', np.mean(acc))
            print('node no is',net.nNodes)
            depth.append(net.nNodes)
            
        #save results to file
        output_file = open(fileName, 'wb')
        results = {'SI':SI,'Selflabel':Self_label,'NormOmega':NormOmega,
                   'Flexiblelayer':Flexible,'Performance':Accuracy,'Nodes':depth}
        pickle.dump(results,output_file)
        output_file.close()
        #save results to file
        metric_file = open(metricfile, 'wb')
        pickle.dump(Metrics,metric_file)
        metric_file.close()
        #generate image
        b = ['Parsnet++']
        # plot(Accuracy,b,picName)

        print('...........results...........')
        print('Average')
        for head,re in zip(b,Accuracy):
            print(head)
            # print(np.mean(re))
            print(np.mean(re[1:]))
        print('F1')
        for i in range(len(SI)):
            print(b[i])
            f1 = []
            for f in Metrics[i]:
                f1.append(f['F1'])
            # print(np.mean(f1))
            print(np.mean(f1[1:]))

    else:
        # print('reulst file exists')
        loadresult(fileName,picName,metricfile)
        if e == (len(seedlist)-1):
            print('average accuracy:\n',np.mean(meanacc))
            a1 = []
            a2 = []
            a3 = []
            for i in range(len(met)):
                a1.append(np.mean(met[i][0]))
                a2.append(np.mean(met[i][1]))
                a3.append(np.mean(met[i][2]))
            print('F1:',np.mean(a1))
            print('Precision:', np.mean(a2))
            print('Recall:', np.mean(a3))
            print(np.std(meanacc))
            print(np.std(a1))
            print(np.std(a2))
            print(np.std(a3))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




