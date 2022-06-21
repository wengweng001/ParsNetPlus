import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score

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


def addGaussianNoise1D(x,std = 0.001, device='cpu'):
    # noiseIntensity: sigma in gaussian noise function
    nData, nFeat = x.shape
    y = x.clone().detach()
    for i, _ in enumerate(x):
        noise = torch.normal(0, std, size = [nFeat,])
        noise = noise.to(device)
        y[i] = x[i] + noise
    return y


class ACM():
    '''
    Autonomous Clustering Mechanism
    '''
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
                    d.append(torch.norm(data - self.center[k]).detach().cpu())
                idx = np.argmin(d)
                dis = min(d)
                k3 = 2 * torch.exp(-dis ** 2) + 1
                cri = np.mean((self.l2d[idx])) + k3 * np.std(self.l2d[idx])  # equ.13 right side

                if dis > cri and self.count > 10 and (i+1) != self.batchSize:  # 10 data points for initialise
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

                if (i+1) % self.batchSize == 0 and self.k > self.classes and grow_cluster == 0:
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

classes = 3 # injection molding dataset
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
        if x[0][0] > 0.6:                   # equ (2)
            h = un_h[k]
            if x[1][0]>2 or x[1][0]<0:
                continue
            predict_acm = acm.pseudo_criterion(h, x[1][0])
            if predict_acm > 0.8:
                cri3 = x[0][0]/(x[0][0]+x[0][1])
                if cri3>0.8:
                    pseu_data.append(h)
                    pseu_ss.append(u_sen[k])
                    pseu_y.append(x[1][0])
                    idx.append(k)
                    # print(x[0][0], predict_acm, cri3)
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