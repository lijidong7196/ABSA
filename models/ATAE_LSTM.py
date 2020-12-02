import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules import Embedding,LSTM,Linear,CrossEntropyLoss

class Attention(Module):
    def __init__(self,hidden_dim_size,apsect_dim_size):
        self.hidden_dim_size = hidden_dim_size
        self.apsect_dim_size = apsect_dim_size
        self.W_h = Linear(self.hidden_dim_size,self.hidden_dim_size)
        self.W_v = Linear(self.apsect_dim_size,self.apsect_dim_size)
        self.w = Linear(1, self.hidden_dim_size + self.apsect_dim_size)
        super(Attention, self).__init__()

    def forward(self,hidden,aspect):
        M = torch.tanh(torch.cat([self.W_h(hidden),self.W_v(aspect)],dim=-1))
        # (1, N) * (d +d_a , N)
        alpha = torch.softmax(self.w)
        torch.cat([])
        pass

class ATAELSTM(Module):

    def __init__(self,input_size,embed_size, hidden_size,aspect_size,embedding=None):
        super(ATAELSTM, self).__init__()
        self.embed_size = embed_size
        self.aspect_size = aspect_size
        if embedding is not None:
            self.embeding = Embedding.from_pretrained(embedding,padding_idx=0)
        else:
            self.embeding = Embedding(input_size,embed_size,padding_idx=0)
        # (batch size, N, embedding size)
        self.apect_embeding = Embedding(aspect_size,embed_size)
        self.rnn = LSTM(input_size=embed_size,
                        hidden_size=hidden_size,
                        bidirectional=True,
                        batch_first=True,
                        num_layers=1)

    def forward(self,input,term):
        x = self.embeding(input)
        aspect = self.apect_embeding(term)
        feature = torch.cat([x,aspect],dim=-1)
        _,() = self.rnn(feature)
    def param_init(self):
        pass


