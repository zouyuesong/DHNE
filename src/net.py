import torch
import torch.nn as nn

from IPython import embed

# a little tricky for layer lists
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class hypergraph_network(nn.Module):
    def __init__(self, opt):
        super(hypergraph_network, self).__init__()
        self.dim_feature = opt.dim_feature
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        print(opt.dim_feature)
        print(opt.embedding_size)

        for i, (input_dim, embedding_dim) in enumerate(zip(opt.dim_feature, opt.embedding_size)):
            self.add_module('encoder_'+str(i), nn.Linear(input_dim, embedding_dim))
            self.add_module('decoder_'+str(i), nn.Linear(embedding_dim, input_dim))
        self.encoder = AttrProxy(self, 'encoder_')
        self.decoder = AttrProxy(self, 'decoder_')
        
        # self.encoders = [nn.Linear(opt.dim_feature[i], opt.embedding_size[i]) for i in range(3)]
        # self.encoder_0 = nn.Linear(opt.dim_feature[0], opt.embedding_size[0])
        # self.encoder_1 = nn.Linear(opt.dim_feature[1], opt.embedding_size[1])
        # self.encoder_2 = nn.Linear(opt.dim_feature[2], opt.embedding_size[2])
        # self.decoders = [nn.Linear(opt.embedding_size[i], opt.dim_feature[i]) for i in range(3)]
        # self.decoder_0 = nn.Linear(opt.embedding_size[0], opt.dim_feature[0])
        # self.decoder_1 = nn.Linear(opt.embedding_size[1], opt.dim_feature[1])
        # self.decoder_2 = nn.Linear(opt.embedding_size[2], opt.dim_feature[2])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.hidden_layer = nn.Linear(sum(opt.embedding_size), opt.hidden_size)
        self.output_layer = nn.Linear(opt.hidden_size, 1)
        
        # FIXME: need to find a way to sequentially express the network
        print(self)


    def forward(self, x, only_embedding=False):
        # encoded_0 = self.tanh(self.encoder_0(x[0]))
        # encoded_1 = self.tanh(self.encoder_1(x[1]))
        # encoded_2 = self.tanh(self.encoder_2(x[2]))
        encoded = [self.encoder[i](x[i]) for i in range(3)]
        encoded = [self.tanh(encoded[i]) for i in range(3)]

        hidden_state = torch.cat(encoded, dim=1)
        hidden_state = self.hidden_layer(hidden_state)
        hidden_state = self.tanh(hidden_state)

        if only_embedding:
            return hidden_state
        else:
            predict = self.output_layer(hidden_state)
            predict = self.sigmoid(predict).flatten()

            # decoded_0 = self.sigmoid(self.decoder_0(encoded_0))
            # decoded_1 = self.sigmoid(self.decoder_1(encoded_1))
            # decoded_2 = self.sigmoid(self.decoder_2(encoded_2))
            decoded = [self.decoder[i](encoded[i]) for i in range(3)]
            decoded = [self.sigmoid(decoded[i]) for i in range(3)]
            return (decoded, predict)

    def get_embedding(self, x, i):
        return self.encoder[i](x)
    


