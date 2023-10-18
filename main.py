from init import loadData, normalize, generater, setup_seed
from train_test import train_test
import torch
import os

dataset = 'river'
windowsize = 5
EPOCHES = 300
batchsize = 32
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    setup_seed(123)
    X1, X2, Y = loadData(dataset)
    X1 = normalize(X=X1, type=2) #选择标准化或者归一化
    X2 = normalize(X=X2, type=2)

    TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter = generater(X_1=X1,
                                                                         X_2=X2,
                                                                         Y=Y,
                                                                         batchsize=batchsize,
                                                                         windowSize=windowsize)
    train_test(
        dataset=dataset,
        train_iter=train_iter,
        test_iter=test_iter,
        TRAIN_SIZE=TRAIN_SIZE,
        TEST_SIZE=TEST_SIZE,
        TOTAL_SIZE=TOTAL_SIZE,
        device=device,
        epoches=EPOCHES,
        windowsize=windowsize)



if __name__ == '__main__':
    main()
