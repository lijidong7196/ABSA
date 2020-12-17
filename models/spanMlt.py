import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from torch.utils.data import DataLoader
import torch.optim as optim


class SpanMltEvaluator():

    def __init__(self, **kwargs):
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = LeNet(classes=5)
        self.net.initialize_weights()
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.cfg.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'norm_mean': [0.485, 0.456, 0.406],
            'norm_std': [0.229, 0.224, 0.225],
            'batch_size': 16,
            'epoch': 50,
            'log_interval': 10,
            'lr': 0.01,
            'num_workers': 8,
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    def test(self, test_dir="Datasets/cartoon_set_test/"):
        self.net.eval()
        # test_dir = "Datasets/celeba_test/"
        net_path = 'B2/model/model.pth'
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                self.cfg.norm_mean, self.cfg.norm_std),
        ])

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
    def train(self, train_dir="Datasets/cartoon_set/", save_dir='B2/model'):

        self.net.train()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.norm_mean, self.cfg.norm_std),
        ])
        train_data = CELEBAData(data_dir=train_dir, transform=train_transform)
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

            test_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.cfg.norm_mean, self.cfg.norm_std),
            ])

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
