import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules import Embedding,LSTM,Linear,CrossEntropyLoss

class Attention(Module):
    def __init__(self,input):
        super(Attention, self).__init__()


class ATAELSTM(Module):

    def __init__(self,input_size,embed_size, hidden_size,aspect_size,embedding=None):
        super(ATAELSTM, self).__init__()
        self.embed_size = embed_size
        self.aspect_size = aspect_size
        if embedding is not None:
            self.embeding = Embedding.from_pretrained(embedding,padding_idx=0)
        else:
            self.embeding = Embedding(input_size,embed_size,padding_idx=0)

        self.apect_embeding = Embedding(aspect_size,embed_size)
        self.rnn = LSTM(input_size=embed_size,
                        hidden_size=hidden_size,
                        bidirectional=True,
                        batch_first=True,
                        num_layers=1)

    def forward(self,input,term):
        x = self.embeding(input)
        aspect = self.apect_embeding(term)
        torch.cat([x,aspect],dim=-1)


    def param_init(self):
        pass



