import numpy as np
import os, sys
import copy
import tensorflow as tf
import argparse
from functools import reduce
import math
import time

import torch
import torch.nn as nn

from dataset import read_datasets, HNDataProvider
from net import hypergraph_network
from utils import save_model, load_model, save_embeddings

from IPython import embed


parser = argparse.ArgumentParser("hyper-network embedding", fromfile_prefix_chars='@')
parser.add_argument('--data_path', type=str, help='Directory to load data.')
parser.add_argument('--save_path', type=str, help='Directory to save data.')
parser.add_argument('-s', '--embedding_size', type=int, nargs=3, default=[32, 32, 32], help='The embedding dimension size')
parser.add_argument('--prefix_path', type=str, default='model', help='.')
parser.add_argument('--hidden_size', type=int, default=64, help='The hidden full connected layer size')
parser.add_argument('-e', '--epochs_to_train', type=int, default=10, help='Number of epoch to train. Each epoch processes the training data once completely')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Number of training examples processed per step')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-a', '--alpha', type=float, default=1, help='radio of autoencoder loss')
parser.add_argument('-neg', '--num_neg_samples', type=int, default=5, help='Neggative samples per training example')
parser.add_argument('-o', '--options', type=str, help='options files to read, if empty, stdin is used')
parser.add_argument('--seed', type=int, help='random seed')


def load_config(config_file):
    with open(config_file, 'r') as f:
        args = parser.parse_args(reduce(lambda a, b: a+b, map(lambda x: ('--'+x).strip().split(), f.readlines())))
    return args

opt = parser.parse_args()
if opt.options is not None:
    opt = load_config(opt.options)
if opt.seed is not None:
    np.random.seed(opt.seed)

# dataset
dataset = read_datasets(opt.data_path)
print(dataset)
train_provider = HNDataProvider(dataset.train, dataset.embeddings, batch_size=opt.batch_size, num_neg_samples=opt.num_neg_samples)
test_provider = HNDataProvider(dataset.test, dataset.embeddings, batch_size=opt.batch_size, num_neg_samples=opt.num_neg_samples)

# net
opt.dim_feature = [sum(dataset.train.nums_type)-n for n in dataset.train.nums_type]
net = hypergraph_network(opt)
# # if need to load model
# load_model(net, opt)

# optimizer
optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.learning_rate)

# loss
BCE_loss = nn.BCELoss()
def sparse_autoencoder_error(y_pred, y_true):
    return ((torch.sign(y_true)*(y_true-y_pred))**2).mean()
MSE = nn.MSELoss()


def Mean(result_list):
    return np.array([r.data.numpy() for r in result_list]).mean()

# process
begin_time = time.time()
for epoch in range(opt.epochs_to_train):
    print("\nepoch:", epoch)
    # train
    it = 0
    loss, dec_loss, cla_loss = [], [[], [], []], []
    mse, acc = [[], [], []], []
    while 1:
        batch = train_provider.next()
        if batch == None:
            break

        x, target, label = batch['embedding'], batch['embedding'], batch['label']
        (decoded, predict) = net(x)
        
        # loss
        dec_loss[0].append(sparse_autoencoder_error(decoded[0], target[0]))
        dec_loss[1].append(sparse_autoencoder_error(decoded[1], target[1]))
        dec_loss[2].append(sparse_autoencoder_error(decoded[2], target[2]))
        cla_loss.append(BCE_loss(predict, label))
        loss.append(sum([dec_loss[i][it] for i in range(3)]) * opt.alpha + cla_loss[it])
        # metrics
        mse[0].append(MSE(decoded[0], target[0]))
        mse[1].append(MSE(decoded[1], target[1]))
        mse[2].append(MSE(decoded[2], target[2]))
        acc.append((predict > 0.5) == label.byte())

        optimizer.zero_grad()
        loss[it].backward()
        optimizer.step()
        it += 1 
    print("     loss: %4f, decode_0: %4f, decode_1: %4f, decode_2: %4f, classify_loss: %4f"
         % (Mean(loss), Mean(dec_loss[0]),Mean(dec_loss[1]), Mean(dec_loss[2]), Mean(cla_loss)))
    print("     mean_square_error: decode_0: %4f, decode_1: %4f, decode_2: %4f"
         % (Mean(mse[0]), Mean(mse[1]), Mean(mse[2])))
    print("     classify_accuracy: %.4f" % Mean(acc))
 
    # test
    net.eval()
    it = 0
    val_loss, val_dec_loss, val_cla_loss = [], [[], [], []], []
    val_mse, val_acc = [[], [], []], []
    while 1:
        batch = test_provider.next()
        if batch == None:
            break

        x, target, label = batch['embedding'], batch['embedding'], batch['label']
        (decoded, predict) = net(x)
        
        # loss
        val_dec_loss[0].append(sparse_autoencoder_error(decoded[0], target[0]))
        val_dec_loss[1].append(sparse_autoencoder_error(decoded[1], target[1]))
        val_dec_loss[2].append(sparse_autoencoder_error(decoded[2], target[2]))
        val_cla_loss.append(BCE_loss(predict, label))
        val_loss.append(sum([val_dec_loss[i][it] for i in range(3)]) * opt.alpha + val_cla_loss[it])
        # metrics
        val_mse[0].append(MSE(decoded[0], target[0]))
        val_mse[1].append(MSE(decoded[1], target[1]))
        val_mse[2].append(MSE(decoded[2], target[2]))
        val_acc.append((predict > 0.5) == label.byte())
        it += 1

    net.train()
    print("   validation: \n       val_loss: %4f, decode_0: %4f, decode_1: %4f, decode_2: %4f, classify_loss: %4f"
         % (Mean(val_loss), Mean(val_dec_loss[0]),Mean(val_dec_loss[1]), Mean(val_dec_loss[2]), Mean(val_cla_loss)))
    print("       val_mean_square_error: decode_0: %4f, decode_1: %4f, decode_2: %4f"
         % (Mean(val_mse[0]), Mean(val_mse[1]), Mean(val_mse[2])))
    print("       val_classify_accuracy: %.4f" % Mean(val_acc))
    

# save model & embeddings
save_model(net, opt)
save_embeddings(net, dataset, opt)

end_time = time.time()
print("time: ", end_time - begin_time)
