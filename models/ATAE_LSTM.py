import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules import Embedding,LSTM,Linear,CrossEntropyLoss

class Attention(Module):
    def __init__(self,hidden_dim_size,apsect_dim_size):
        super(Attention, self).__init__()
        self.hidden_dim_size = hidden_dim_size
        self.apsect_dim_size = apsect_dim_size
        #(d,d)
        self.W_h = Linear(self.hidden_dim_size,self.hidden_dim_size)
        #(d_a,d_a)
        self.W_v = Linear(self.apsect_dim_size,self.apsect_dim_size)
        #(1, d_a)
        self.w = Linear(self.hidden_dim_size + self.apsect_dim_size,1)

    def forward(self,hidden,aspect):
        # transform the aspect embedding into shape of (batch, 1, v_a)
        transfored_v = self.W_v(aspect)
        # repeated the the transfored aspect embedding for N times to generate tensor with shape of (batch, N, v_a)
        repeated_v = transfored_v.repeat(1,hidden.size(1),1)
        # now, we concat transformed hidden and aspect embedding vector to generate M tensor with shape of (batch, N,d+d_a)
        M = torch.tanh(torch.cat([self.W_h(hidden),repeated_v],dim=-1))
        # then, we firstly transformed M with the matrix w, then pass it into softmax function
        # to generate shape of (batch, N, 1)
        alpha = torch.softmax(self.w(M).squeeze(-1),dim=-1).unsqueeze(-1)
        # we get the weight for each hidden, then, we attach it to hidden state and generate the shape of (batch, 1, v)
        r = torch.bmm(alpha.permute(0,2,1),hidden)

        return  r


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
        self.att = Attention(hidden_size,aspect_size)

    def forward(self,input,term):
        x = self.embeding(input)
        aspect = self.apect_embeding(term)
        output,_ = self.rnn(x)
        r = self.att(output,aspect)

    def param_init(self):
        pass





