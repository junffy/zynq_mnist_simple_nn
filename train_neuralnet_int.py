# This code is based on 
# https://github.com/oreilly-japan/deep-learning-from-scratch
# http://marsee101.blog19.fc2.com
# This modified code also takes over the MIT License.

# coding: utf-8
import sys, os
sys.path.append("./")

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_int import TwoLayerNet
from cpp.fwrite_weight import fwrite_weight
from cpp.fwrite_bias import fwrite_bias
from cpp.view_mnist import view_mnist
# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
        test_mask = np.random.choice(train_size, 1)
        xtest_batch = x_train[test_mask]
        ttest_batch = t_train[test_mask]
        xtest_data = network.predict(xtest_batch)
        #print(train_acc, test_acc)
        #print(xtest_data)
        #print(ttest_batch)
# print(network.params['W1'])
# print(network.params['b1'])
# print(network.params['W2'])
# print(network.params['b2'])

MAGNIFICATION = 2 ** (9-1)
fwrite_weight(network.params['W1'], 'af1_weight.h', 'af1_fweight', 'af1_weight', MAGNIFICATION, 784, 50)
fwrite_weight(network.params['W2'], 'af2_weight.h', 'af2_fweight', 'af2_weight', MAGNIFICATION, 50, 10)
fwrite_bias(network.params['b1'], 'af1_bias.h', 'af1_fbias', 'af1_bias', MAGNIFICATION, 50)
fwrite_bias(network.params['b2'], 'af2_bias.h', 'af2_fbias', 'af2_bias', MAGNIFICATION, 10)


print()
train_acc_int = network.accuracy_int(x_train, t_train)
test_acc_int = network.accuracy_int(x_test, t_test)
print(train_acc_int, test_acc_int)
