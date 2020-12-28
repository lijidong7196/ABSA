import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules import Embedding,LSTM,Linear,CrossEntropyLoss
import os
import random
import numpy as np
import torch
from collections import namedtuple
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from utils.data_hepler import ABSADataset

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
        # to define projection parameters for W_p and W_x
        self.W_p = Linear(self.hidden_dim_size,self.hidden_dim_size)
        self.W_x = Linear(self.hidden_dim_size,self.hidden_dim_size)


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
        # we get the weight for each hidden units and Then we attach them to
        output = torch.tanh(self.W_p(r) + self.W_x(hidden[:,-1,:]))
        return  output

class ATAELSTM(Module):

    def __init__(self,input_size,embed_size, hidden_size,aspect_size,num_class,embedding=None):
        super(ATAELSTM, self).__init__()
        self.embed_size = embed_size
        self.aspect_size = aspect_size
        self.num_class = num_class
        # emeddding 
        if embedding is not None:
            self.embeding = Embedding.from_pretrained(embedding,padding_idx=0)
            self.embeding.weight.requires_grad = False
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
        self.fc = Linear(hidden_size,num_class,bias=True)

    def forward(self,input,term):
        x = self.embeding(input)
        aspect = self.apect_embeding(term)
        output,_ = self.rnn(x)
        # output the final attention represent
        att_present = self.att(output,aspect).squeeze(1)
        output = torch.softmax(att_present,dim=-1)
        return output


class SpanMltEvaluator():

    def __init__(self, **kwargs):
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = ATAELSTM(50000,300,300,300,20)
        #self.net.initialize_weights()
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adagrad(
            self.net.parameters(), lr=self.cfg.lr,weight_decay=self.cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'norm_mean': [0.485, 0.456, 0.406],
            'norm_std': [0.229, 0.224, 0.225],
            'batch_size': 25,
            'epoch': 50,
            'log_interval': 10,
            'lr': 0.01,
            'num_workers': 8,
            'weight_decay':0.001
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    def test(self, test_dir="Datasets/cartoon_set_test/"):
        self.net.eval()
        # test_dir = "Datasets/celeba_test/"
        net_path = 'B2/model/model.pth'


        test_data = CELEBAData(
            data_dir=test_dir, transform=test_transform)
        valid_loader = DataLoader(dataset=test_data, batch_size=1)
        self.net.load_state_dict(torch.load(
            net_path, map_location=lambda storage, loc: storage))
        total_t = 0
        correct_t = 0

        for i, data in enumerate(valid_loader):
            # forward
            inputs, labels = data
            inputs = inputs.to(self.device, non_blocking=self.cuda)
            labels = labels.to(self.device, non_blocking=self.cuda)
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_t += 1
            correct_t += (predicted ==
                          labels).squeeze().sum().cpu().numpy()
        acc = correct_t/total_t
        print("Eye_color识别率为%.2f" % (acc))

    @ torch.enable_grad()
    def train(self, train_dir, save_dir='save_models'):

        self.net.train()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            shuffle=True)
        max_acc = 0
        for epoch in range(self.cfg.epoch):

            loss_train = 0.
            for i, data in enumerate(train_loader):

                # forward
                inputs, labels = data
                inputs = inputs.to(self.device, non_blocking=self.cuda)
                labels = labels.to(self.device, non_blocking=self.cuda)
                outputs = self.net(inputs)

                # backward
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()

                # update weights
                self.optimizer.step()

                loss_train += loss.item()
                if (i+1) % self.cfg.log_interval == 0:
                    loss_mean = loss_train / self.cfg.log_interval
                    print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                        epoch+1, self.cfg.epoch, i+1, len(train_loader), loss_mean))
                    loss_train = 0.
            test_dir = "Datasets/cartoon_set_test/"

            test_data = CELEBAData(
                data_dir=test_dir, transform=test_transform)
            valid_loader = DataLoader(dataset=test_data, batch_size=1)
            total_t = 0
            correct_t = 0

            for i, data in enumerate(valid_loader):
                # forward
                inputs, labels = data
                inputs = inputs.to(self.device, non_blocking=self.cuda)
                labels = labels.to(self.device, non_blocking=self.cuda)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_t += 1
                correct_t += (predicted ==
                              labels).squeeze().sum().cpu().numpy()
            acc = correct_t/total_t
            print("识别率为%.2f" % (acc))

            self.scheduler.step()
            if max_acc < acc:
                max_acc = acc
                net_path = os.path.join(
                    save_dir, 'model.pth')
                torch.save(self.net.state_dict(), net_path)


if __name__ == "__main__":
    eye_color = Eye_color()
    eye_color.train("../Datasets/cartoon_set/")
