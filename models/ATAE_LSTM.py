import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules import Embedding,LSTM,Linear,CrossEntropyLoss

class Attention(Module):
    def __init__(self,input):
        super(Attention, self).__init__()


class ATAELSTM(Module):

    def __init__(self,input_size,embed_size, batch_size, hidden_size,embedding=None):
        super(ATAELSTM, self).__init__()
        self.embed_size = embed_size
        if embedding is not None:
            self.embeding = Embedding.from_pretrained(embedding,padding_idx=0)
        else:
            self.embeding = Embedding(input_size,embed_size,padding_idx=0)

        self.rnn = LSTM(input_size=embed_size,hidden_size=hidden_size,bidirectional=True,batch_first=True,num_layers=1)
        self.rnn = LSTM()


    def forward(self,input,term):
        pass

    def param_init(self):
        pass



