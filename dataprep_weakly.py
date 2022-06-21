import torch
from torch.utils.data import Dataset
import numpy as np
import os
from scipy import io

def get_labeled_index(labeled_rate):
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    idx = np.arange(2952)    # the dataset has 2952 data
    np.random.seed(6)
    np.random.shuffle(idx)
    train_labeled_idxs.extend(idx[:int(2952*labeled_rate)])
    train_unlabeled_idxs.extend(idx[int(2952*labeled_rate):])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs

def get_labeled_index_next(labeled_rate):
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    idx = np.arange(2902)    # the dataset has 2952 data
    np.random.seed(2)
    np.random.shuffle(idx)
    train_labeled_idxs.extend(idx[:int(2902*labeled_rate)])
    train_unlabeled_idxs.extend(idx[int(2902*labeled_rate):])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs

def sensor(next=False):
    data1 = io.loadmat(r'data/injordexp2952.mat')
    data  = data1.get('data')
    data  = torch.from_numpy(data)
    data  = data.float()
    preq_data = data[:,0:-1]
    preq_label        = data[:,-1]
    if next:
        nData             = preq_data.shape[0]
        dataShift         = 50
        preq_data         = preq_data[0:nData - dataShift]
        preq_label        = preq_label[dataShift:]
    nInputSensor      = preq_data.shape[1]
    return preq_data,preq_label,nInputSensor


class sensordataset(Dataset):
    def __init__(self, labeled=True, indexs=None, next=False):
        samples, targets, _ = sensor(next)
        self.samples = samples
        self.targets = targets
        if labeled:
            if indexs is not None:
                self.samples = samples[indexs]
                self.targets = targets[indexs]
        else:
            if indexs is not None:
                self.samples = samples[indexs]
                self.targets = targets[indexs]

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

    def __len__(self):
        return self.samples.size(0)

def load_sensor(labeled_proportion, batchsize, next=False):
    '''

    Args:
        labeled_proportion: dataset labelled rate
        batchsize: mini batch size
        next:   True--Next batch prediction dataset
                False--Current batch prediction dataset

    Returns:
        labeled_dataloader:     labelled dataloader
        unlabeled_dataloader:   unlabelled dataloader

    '''
    # load sensor data
    if os.path.isfile('./data/label_idx_rate{}.npy'.format(labeled_proportion)) and os.path.isfile('./data/unlabel_idx_rate{}.npy'.format(labeled_proportion)):
        print("loading idx")
        train_labeled_idxs = np.load('./data/label_idx_rate{}.npy'.format(labeled_proportion))
        train_unlabeled_idxs = np.load('./data/unlabel_idx_rate{}.npy'.format(labeled_proportion))
    else:
        train_labeled_idxs, train_unlabeled_idxs = get_labeled_index(labeled_rate=labeled_proportion)
        np.save('./data/label_idx_rate{}.npy'.format(labeled_proportion),train_labeled_idxs)
        np.save('./data/unlabel_idx_rate{}.npy'.format(labeled_proportion),train_unlabeled_idxs)
    if next:
        train_labeled_idxs, train_unlabeled_idxs = get_labeled_index_next(labeled_rate=labeled_proportion)
    train_labeled_dataset = sensordataset(labeled=True, indexs=train_labeled_idxs, next=next)
    train_unlabeled_dataset = sensordataset(labeled=False, indexs=train_unlabeled_idxs, next=next)

    labeled_dataloader = torch.utils.data.DataLoader(
        train_labeled_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    unlabeled_dataloader = torch.utils.data.DataLoader(
        train_unlabeled_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    print(f"#Labeled sensor: {len(train_labeled_idxs)} #Unlabeled sensor: {len(train_unlabeled_idxs)}")

    return labeled_dataloader, unlabeled_dataloader