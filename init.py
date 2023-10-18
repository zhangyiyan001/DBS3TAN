# -*- coding:utf-8 -*-

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from operator import truediv
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

import torch.utils.data as Data
import torch.optim.lr_scheduler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loadData(names):
    if names == 'river':
        data_path = os.path.join(r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset\datasets')
        data1 = sio.loadmat(os.path.join(data_path, 'river_before.mat'))['river_before']
        data2 = sio.loadmat(os.path.join(data_path, 'river_after.mat'))['river_after']
        labels = sio.loadmat(os.path.join(data_path, 'groundtruth.mat'))['lakelabel_v1']
        labels[labels == 255] = 1
        # labels = sio.loadmat(os.path.join(data_path, 'groundtruth.mat')).keys()
        # print(labels)
    if names == 'farm':
        data_path = os.path.join(r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset\datasets')
        data1 = sio.loadmat(os.path.join(data_path, 'farm06.mat'))['imgh']
        data2 = sio.loadmat(os.path.join(data_path, 'farm07.mat'))['imghl']
        labels = sio.loadmat(os.path.join(data_path, 'label.mat'))['label']

    if names == 'Hermiston':
        data_path = os.path.join(r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset\datasets')
        data1 = sio.loadmat(os.path.join(data_path, 'hermiston2004.mat'))['HypeRvieW']
        data2 = sio.loadmat(os.path.join(data_path, 'hermiston2007.mat'))['HypeRvieW']
        labels = sio.loadmat(os.path.join(data_path, 'label.mat'))['label']


    return data1, data2, labels

def set_train_sample(x, y, pos, neg):
    rand_perm = np.random.permutation(y.shape[0])
    new_x = x[rand_perm, :, :, :]
    new_y = y[rand_perm]

    train_x0 = new_x[new_y == 0, :, :, :][:neg]
    train_y0 = new_y[new_y == 0][:neg]
    train_x1 = new_x[new_y == 1, :, :, :][:pos]
    train_y1 = new_y[new_y == 1][:pos]

    test_x0 = new_x[new_y == 0, :, :, :][neg:]
    test_y0 = new_y[new_y == 0][neg:]
    test_x1 = new_x[new_y == 1, :, :, :][pos:]
    test_y1 = new_y[new_y == 1][pos:]

    x_train = np.concatenate((train_x0, train_x1))
    y_train = np.concatenate((train_y0, train_y1))
    x_test = np.concatenate((test_x0, test_x1))
    y_test = np.concatenate((test_y0, test_y1))
    return  x_train, x_test, y_train, y_test


def split_data(x, y, pos, neg):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    selected_pos_indices = np.random.choice(pos_indices, pos, replace=False)
    selected_neg_indices = np.random.choice(neg_indices, neg, replace=False)

    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))

    x_train = x[selected_indices]
    y_train = y[selected_indices]

    mask = np.ones(y.shape[0], dtype=bool)
    mask[selected_indices] = False
    x_test = x[mask]
    y_test = y[mask]

    return x_train, x_test, y_train, y_test


#划分path块的方法
def pad_with_zeros(X, margin):
    """apply zero padding to X with margin"""

    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return new_X


def create_patches(X, y, window_size, remove_zero_labels=True):
    """create patch from image. suppose the image has the shape (w,h,c) then the patch shape is
    (w*h,window_size,window_size,c)"""

    margin = int((window_size - 1) / 2)
    zero_padded_X = pad_with_zeros(X, margin=margin)
    # split patches
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patchs_labels = np.zeros((X.shape[0] * X.shape[1]))
    patch_index = 0
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            patch = zero_padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index, :, :, :] = patch
            patchs_labels[patch_index] = y[r - margin, c - margin] + 1
            patch_index = patch_index + 1

    if remove_zero_labels:
        patches_data = patches_data[patchs_labels > 0, :, :, :]
        patchs_labels = patchs_labels[patchs_labels > 0]
        patchs_labels -= 1
    return patches_data, patchs_labels


def normalize(X, type):
    x = np.zeros(shape=X.shape, dtype='float32')
    if type == 1:
        for i in range(X.shape[2]):
            temp = X[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)
    if type == 2:
        for i in range(X.shape[2]):
            min = np.min(X[:, :, i])
            max = np.max(X[:, :, i])
            scale = max - min
            if scale == 0:
                scale = 1e-5
            x[:, :, i] = (X[:, :, i] - min) / scale
    return x


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)  # 获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)  #
    return np.round(each_acc, 4), average_acc


def reports(test_iter, net, device):
    y_test = []
    y_pred = []
    with torch.no_grad():
        for step, (x1, x2, x3, x4, y) in enumerate(test_iter):
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            y = y.to(device)
            net.eval()
            y_hat1, y_hat2, y_hat3, y_hat4, y_hat = net(x1, x2, x3, x4)
            size = y_hat.numel()
            y_hat = binary(y_hat, th=0.9, bathsize=size)
            y_hat = y_hat.cpu()
            y_hat = y_hat.tolist()
            y_pred.extend(y_hat)
            y_test.extend(y.cpu())
            net.train()
    classification = classification_report(np.array(y_test), np.array(y_pred), digits=2)
    oa = accuracy_score(np.array(y_test), np.array(y_pred))  # 计算OA
    confusion = confusion_matrix(np.array(y_test), np.array(y_pred))  # 计算confusion
    each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算each_acc和aa
    kappa = cohen_kappa_score(np.array(y_test), np.array(y_pred))  # 计算kappa
    f1 = f1_score(np.array(y_test), np.array(y_pred))
    auc = roc_auc_score(np.array(y_test), np.array(y_pred))

    return classification, confusion, oa * 100, aa * 100, kappa * 100, f1 * 100, auc * 100


def generater(X_1, X_2, Y, batchsize, windowSize):
    alldataX1, alldataY = create_patches(X_1, Y, window_size=windowSize)
    alldataX1 = np.transpose(alldataX1, (0, 3, 1, 2))
    X1_train, X1_test, y_train, y_test = set_train_sample(alldataX1, alldataY, pos=1250, neg=2500)
    x1_train_vit = X1_train.reshape(X1_train.shape[0], windowSize * windowSize, X1_train.shape[1])
    x1_test_vit = X1_test.reshape(X1_test.shape[0], windowSize * windowSize, X1_test.shape[1])

    alldataX2, alldataY = create_patches(X_2, Y, window_size=windowSize)
    alldataX2 = np.transpose(alldataX2, (0, 3, 1, 2))
    X2_train, X2_test, y_train, y_test = set_train_sample(alldataX2, alldataY, pos=1250, neg=2500)
    x2_train_vit = X2_train.reshape(X2_train.shape[0], windowSize * windowSize, X2_train.shape[1])
    x2_test_vit = X2_test.reshape(X2_test.shape[0], windowSize * windowSize, X2_test.shape[1])

    ALL_SIZE = alldataX1.shape[0]
    TRAIN_SIZE = X1_train.shape[0]
    TEST_SIZE = X1_test.shape[0]

    X1_train_tensor = torch.from_numpy(X1_train).type(torch.FloatTensor)
    X2_train_tensor = torch.from_numpy(X2_train).type(torch.FloatTensor)
    Y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    X1_train_vit_tensor = torch.from_numpy(x1_train_vit).type(torch.FloatTensor)
    x2_train_vit_tensor = torch.from_numpy(x2_train_vit).type(torch.FloatTensor)

    X1_test_tensor = torch.from_numpy(X1_test).type(torch.FloatTensor)
    X2_test_tensor = torch.from_numpy(X2_test).type(torch.FloatTensor)
    Y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor)
    x1_test_vit_tensor = torch.from_numpy(x1_test_vit).type(torch.FloatTensor)
    x2_test_vit_tensor = torch.from_numpy(x2_test_vit).type(torch.FloatTensor)



    torch_train = Data.TensorDataset(X1_train_tensor, X2_train_tensor, X1_train_vit_tensor, x2_train_vit_tensor,
                                     Y_train_tensor)
    torch_test = Data.TensorDataset(X1_test_tensor, X2_test_tensor, x1_test_vit_tensor, x2_test_vit_tensor,
                                    Y_test_tensor)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0
    )


    return TRAIN_SIZE, TEST_SIZE, ALL_SIZE, train_iter, test_iter


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def weight_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

def l2_penalty(w):
    return (w ** 2).sum / 2

def binary(x, th, bathsize):
    a = torch.zeros([bathsize])
    a = a.cuda()
    b = torch.ones([bathsize])
    b = b.cuda()
    x = torch.where(x <= th, x, b)
    x = torch.where(x > th, x, a)
    return x