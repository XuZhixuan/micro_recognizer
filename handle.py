import torch
from torch import save
from torch.utils.data.dataloader import DataLoader

from container import Container
from helper import *


class Handler:
    def __init__(self, app: Container):
        self.train_set = None
        self.validate_set = None
        self.app = app

    def create_dataset(self):
        train_set, validate_set = train_test_split(self.app.source, 0.1)
        self.train_set = DataLoader(train_set, batch_size=4)
        self.validate_set = DataLoader(validate_set, batch_size=4)

    def summary(self, epoch: int, pred: list):
        save(self.app.model, './storage/bin/' + self.app.helper.time_name() + '/model-' + str(epoch) + '.pt')
        self.app.helper.dump_json(
            './storage/logs/summary/' + self.app.helper.time_name() + '/pred-' + str(epoch) + '.json',
            pred
        )

    def train_network(self, epochs):
        import os
        os.mkdir('./storage/bin/' + self.app.helper.time_name())
        os.mkdir('./storage/logs/summary/' + self.app.helper.time_name())
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)
            self.app.model.train()
            train_loss = 0.
            train_correct = 0
            for _, datum in enumerate(self.train_set):
                x, y = datum
                out = self.app.model(x.cuda())
                out = torch.softmax(out, 1)
                loss = self.app.loss_function(out, y.squeeze())
                _, out = torch.max(out, 1)
                train_loss += loss.data.item()
                train_correct += torch.eq(y, out).sum().item()

                # Optimize the params in network
                self.app.optimizer.zero_grad()
                loss.backward()
                self.app.optimizer.step()

            train_loss /= len(self.train_set)
            train_acc = train_correct / len(self.train_set)
            # Log the train loss for tensorboard
            self.app.train_summary.add_scalar('Train_Loss', train_loss, epoch)
            self.app.train_summary.add_scalar('Train_acc', train_acc, epoch)
            print('Train finished, loss=%f, acc=%f' % (train_loss, train_acc), end=' ')

            # Start validating
            self.app.model.eval()
            val_loss = 0.
            val_correct = 0
            pred = []
            for _, datum in enumerate(self.validate_set):
                x, y = datum
                out = self.app.model(x.cuda())
                out = torch.softmax(out, 1)
                loss = self.app.loss_function(out, y.squeeze())
                _, out = torch.max(out, 1)
                pred.extend(out.tolist())
                val_loss += loss.data.item()
                val_correct += torch.eq(out, y).sum().item()

            val_loss /= len(self.validate_set)
            val_acc = val_correct / len(self.validate_set)
            print('Test finished, loss=%f, acc=%f' % (val_loss, val_acc))

            # Log the validate loss & r2 for tensorboard
            self.app.train_summary.add_scalar('Test_acc', val_acc, epoch)
            self.app.train_summary.add_scalar('Test_Loss', val_loss, epoch)

            self.summary(epoch, pred)

    def run(self):
        self.app.network_summary(self.app.model, (1, 256, 256))
        self.create_dataset()
        self.train_network(self.app.config('training.epochs'))
