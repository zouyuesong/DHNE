import torch
import torch.nn as nn

from IPython import embed

class hypergraph_network(nn.Module):
    def __init__(self, opt):
        super(hypergraph_network, self).__init__()
        self.dim_feature = opt.dim_feature
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        print(opt.dim_feature)
        print(opt.embedding_size)
        # self.encoders = [nn.Linear(opt.dim_feature[i], opt.embedding_size[i]) for i in range(3)]
        self.encoder_0 = nn.Linear(opt.dim_feature[0], opt.embedding_size[0])
        self.encoder_1 = nn.Linear(opt.dim_feature[1], opt.embedding_size[1])
        self.encoder_2 = nn.Linear(opt.dim_feature[2], opt.embedding_size[2])
        self.tanh = nn.Tanh()
        # self.decoders = [nn.Linear(opt.embedding_size[i], opt.dim_feature[i]) for i in range(3)]
        self.decoder_0 = nn.Linear(opt.embedding_size[0], opt.dim_feature[0])
        self.decoder_1 = nn.Linear(opt.embedding_size[1], opt.dim_feature[1])
        self.decoder_2 = nn.Linear(opt.embedding_size[2], opt.dim_feature[2])
        self.sigmoid = nn.Sigmoid()

        self.hidden_layer = nn.Linear(sum(opt.embedding_size), opt.hidden_size)
        self.output_layer = nn.Linear(opt.hidden_size, 1)
        
        # TODO: find a way to express the network
        # print(self)


    def forward(self, x, only_embedding=False):
        encoded_0 = self.tanh(self.encoder_0(x[0]))
        encoded_1 = self.tanh(self.encoder_1(x[1]))
        encoded_2 = self.tanh(self.encoder_2(x[2]))
        # encoded = [self.encoders[i](x[i]) for i in range(3)]
        # encoded = [self.tanh(encoded[i]) for i in range(3)]

        hidden_state = torch.cat((encoded_0, encoded_1, encoded_2), dim=1)
        hidden_state = self.hidden_layer(hidden_state)
        hidden_state = self.tanh(hidden_state)

        if only_embedding:
            return hidden_state
        else:
            predict = self.output_layer(hidden_state)
            predict = self.sigmoid(predict).flatten()

            decoded_0 = self.sigmoid(self.decoder_0(encoded_0))
            decoded_1 = self.sigmoid(self.decoder_1(encoded_1))
            decoded_2 = self.sigmoid(self.decoder_2(encoded_2))
            # decoded = [self.decoders[i](encoded[i]) for i in range(3)]
            # decoded = [self.sigmoid(decoded[i]) for i in range(3)]
            return ([decoded_0, decoded_1, decoded_2], predict)


