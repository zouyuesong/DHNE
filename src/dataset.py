import numpy as np
import os
import copy
import collections
import scipy.io as sio
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import sys
import itertools

import torch
import torch.utils.data as Data

from IPython import embed

Datasets = collections.namedtuple('Datasets', ['train', 'test', 'embeddings', 'node_cluster',
                                                'labels', 'idx_label', 'label_name'])

class HNDataset(Data.Dataset):
    def __init__(self, edge, nums_type):
        self.edge = edge
        self.edge_set = set(map(tuple, edge)) ### ugly code, need to be fixed
        self.nums_type = nums_type
        self.nums_examples = len(edge)
        # self.epochs_completed = 0
        # self.index_in_epoch = 0

    def __getitem__(self, idx):
        return self.edge[idx]

    def __len__(self):
        return self.nums_examples



# FIXME: test also be shuffled??
class HNDataProvider:
    def __init__(self, dataset, embeddings, batch_size=16, shuffle=True, drop_last=True, num_neg_samples=1, pair_radio=0.9, sparse_input=True):
        self.dataset = dataset
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_neg_samples = num_neg_samples
        self.pair_radio = pair_radio
        self.sparse_input =sparse_input

        # self.is_cuda = None
        self.dataiter = None
        self.epoch = 0
        self.iteration = 0
    
    def build(self):
        dataloader = Data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)
        self.dataiter = dataloader.__iter__()

    def get_batch(self, pos_data): 
        #  add negative sample
        neg_data = []
        for i in pos_data:
            n_neg = 0
            while n_neg < self.num_neg_samples:
                index = copy.deepcopy(i)
                mode = np.random.rand()
                if mode < self.pair_radio:
                    type_ = np.random.randint(3)
                    node = np.random.randint(self.dataset.nums_type[type_])
                    index[type_] = node
                else:
                    types_ = np.random.choice(3, 2, replace=False)
                    node_1 = np.random.randint(self.dataset.nums_type[types_[0]])
                    node_2 = np.random.randint(self.dataset.nums_type[types_[1]])
                    index[types_[0]] = node_1
                    index[types_[1]] = node_2
                if tuple(index) in self.dataset.edge_set:
                    continue
                n_neg += 1
                neg_data.append(index)
        if len(neg_data) > 0:
            neg_data = torch.stack(neg_data, dim=0)
            batch_data = torch.cat((pos_data, neg_data), dim=0)
            nums_batch = len(batch_data)
            labels = np.zeros(nums_batch)
            labels[0:self.batch_size] = 1
            perm = np.random.permutation(nums_batch)
            batch_data = batch_data[perm]
            labels = labels[perm]
        else:
            batch_data = pos_data
            nums_batch = len(batch_data)
            labels = np.ones(len(batch_data))

        # embedding
        batch_e = embedding_lookup(self.embeddings, batch_data, self.sparse_input)
        return dict([('embedding', [torch.from_numpy(batch_e[i]).float() for i in range(3)]),
                    ('label', torch.from_numpy(labels).float())])

    def next(self):
        """
            Return the next `batch_size` examples from this data set.
            if num_neg_samples = 0, there is no negative sampling.
        """
        if self.dataiter is None:
            self.iteration = 1
            self.build()
        try: 
            pos_data = self.dataiter.next()
            batch = self.get_batch(pos_data)
            self.iteration += 1
            return batch

        except StopIteration:
            self.epoch += 1
            self.dataiter = None
            return None


def embedding_lookup(embeddings, index, sparse_input=True):
    if sparse_input:
        return [embeddings[i][index[:, i], :].todense()  for i in range(3)]
    else:
        return [embeddings[i][index[:, i], :]  for i in range(3)]

def read_datasets(train_dir):
    TRAIN_FILE = 'train_data.npz'
    TEST_FILE = 'test_data.npz'
    data = np.load(os.path.join(train_dir, TRAIN_FILE))
    train_data = HNDataset(data['train_data'], data['nums_type'])
    labels = data['labels'] if 'labels' in data else None
    idx_label = data['idx_label'] if 'idx_label' in data else None
    label_set = data['label_name'] if 'label_name' in data else None
    del data
    data = np.load(os.path.join(train_dir, TEST_FILE))
    test_data = HNDataset(data['test_data'], data['nums_type'])
    node_cluster = data['node_cluster'] if 'node_cluster' in data else None
    test_labels = data['labels'] if 'labels' in data else None
    del data
    embeddings = generate_embeddings(train_data.edge, train_data.nums_type)
    return Datasets(train=train_data, test=test_data, embeddings=embeddings, node_cluster=node_cluster,
                labels=labels, idx_label=idx_label, label_name=label_set)

def generate_H(edge, nums_type):
    nums_examples = len(edge)
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(nums_examples))), shape=(nums_type[i], nums_examples)) for i in range(3)]
    return H

# FIXME: where to use??
def dense_to_onehot(labels):
    return np.array(map(lambda x: [x*0.5+0.5, x*-0.5+0.5], list(labels)), dtype=float)

def generate_embeddings(edge, nums_type, H=None):
    if H is None:
        H = generate_H(edge, nums_type)
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)]
    ### 0-1 scaling
    for i in range(3):
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        _, col_index = embeddings[i].nonzero()
        embeddings[i].data /= col_max[col_index]
    return embeddings
