# -*- coding:utf-8 -*-
import time
import numpy as np
import torch.optim.lr_scheduler
from init import reports, weight_init,binary
import torch.optim as optim
from network import Net

from network import ContrastiveLoss
from torch.optim import lr_scheduler

def train_test(
        dataset,
        train_iter,
        test_iter,
        TRAIN_SIZE,
        TEST_SIZE,
        TOTAL_SIZE,
        device,
        epoches,
        windowsize):
    train_loss_list = []
    net = Net(in_cha=198, patch=windowsize, num_class=2).to(device)
    net.apply(weight_init)  # 网络权重初始化
    optimizer = optim.Adam(net.parameters(), lr=0.0005)  # PU SGD 5e-2
    lr_adjust = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    loss = ContrastiveLoss()
    print('TORAL_SIZE: ', TOTAL_SIZE)
    print('TRAIN_SIZE: ', TRAIN_SIZE)
    print('TEST_SIZE: ', TEST_SIZE)
    print('---Training on {}---\n'.format(device))
    start = time.time()
    for epoch in range(epoches):
        train_loss_sum = 0.0
        time_epoch = time.time()
        for step, (X1, X2, X3, X4, y) in enumerate(train_iter):
            x1 = X1.to(device)
            x2 = X2.to(device)
            x3 = X3.to(device)
            x4 = X4.to(device)
            y = y.to(device)
            y_hat1, y_hat2, y_hat3, y_hat4, y_hat = net(x1, x2, x3, x4)
            l1 = loss(y_hat1, y_hat2, y.long())
            l2 = loss(y_hat3, y_hat4, y.long())
            l = 0.5 * l1 + 0.5 * l2
            optimizer.zero_grad()  # 梯度清零
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
        print('epoch %d, train loss %.6f, time %.2f sec' % (epoch + 1,
                                                            train_loss_sum / len(train_iter.dataset),
                                                            time.time() - time_epoch))

        train_loss_list.append(train_loss_sum / len(train_iter.dataset))
        if train_loss_list[-1] <= min(train_loss_list):
            torch.save(net.state_dict(), './models/' + dataset + '.pt')
            print('***Successfully Saved Best models parametres!***\n')  # 保存在训练集上损失值最好的模型效果
        if epoch % 50==0:
            print('\n***Start  Testing***\n')
            evaluate(test_iter=test_iter, model=net, device=device)
    End = time.time()
    print('***Training End! Total Time %.1f sec***' % (End - start))


def evaluate(test_iter, model, device):
    classification, confusion, oa, aa, kappa, f1, auc = reports(test_iter, model, device=device)
    classification = str(classification)
    confusion = str(confusion)
    print(classification, confusion, oa, aa, kappa, f1, auc)



