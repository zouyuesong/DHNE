import torch
import numpy as np
import os
import math

def get_embeddings(net, dataset, opt):
    # shift = np.append([0], np.cumsum(dataset.train.nums_type))
    embeddings = []
    for i in range(3):
        index = range(dataset.train.nums_type[i])
        batch_num = math.ceil(1.0 * len(index) / opt.batch_size)
        ls = np.array_split(index, batch_num)
        ps = []
        for j, lss in enumerate(ls):
            embed = net.get_embedding(torch.from_numpy(dataset.embeddings[i][lss, :].todense()).float(), i)
            ps.append(embed.data.numpy())
        ps = np.vstack(ps)
        embeddings.append(ps)
    # print(embeddings)
    return embeddings

def save_embeddings(net, dataset, opt, file_name='embeddings.npy'):
    emds = get_embeddings(net, dataset, opt)
    prefix = '{}_{}'.format(opt.prefix_path, opt.embedding_size[0])
    prefix_path = os.path.join(opt.save_path, prefix)
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    np.save(open(os.path.join(prefix_path, file_name), 'wb'), emds)


def save_model(net, opt):
    prefix = '{}_{}'.format(opt.prefix_path, opt.embedding_size[0])
    prefix_path = os.path.join(opt.save_path, prefix)
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    # model
    torch.save(net.state_dict(), os.path.join(prefix_path, 'model_params.pkl'))
    # config
    with open(os.path.join(prefix_path, 'config.txt'), 'w') as f:
        for key, value in vars(opt).items():
            if value is None:
                continue
            if type(value) == list:
                s_v = " ".join(list(map(str, value)))
            else:
                s_v = str(value)
            f.write(key+" "+s_v+'\n')

def load_model(net, opt):
    prefix_path = os.path.join(opt.save_path, '{}_{}'.format(opt.prefix_path, opt.embedding_size[0]))
    net.load_state_dict(torch.load(os.path.join(prefix_path, 'model_params.pkl')))
    # self.model = load_model(, custom_objects={'sparse_autoencoder_error': self.sparse_autoencoder_error})