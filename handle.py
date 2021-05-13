import ignite.contrib.metrics.regression as reg
import matplotlib.pyplot as plot
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

    def summary(self, epoch: int, y_data: list, pred: list):
        save(self.app.model, './storage/bin/' + self.app.helper.time_name() + '/model-' + str(epoch) + '.pt')
        plot.plot(y_data, pred)
        plot.savefig('./storage/logs/' + self.app.helper.time_name() + str(epoch) + '.png')

    def train_network(self, epochs):
        import os
        os.mkdir('./storage/logs/summary/' + self.app.helper.time_name())
        os.mkdir('./storage/bin/' + self.app.helper.time_name())
        y_tmp = list(map(lambda k: k[1].tolist(), self.validate_set))
        y_data = [item for batch in y_tmp for item in batch]
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)
            self.app.model.train()
            train_loss = 0.
            for _, datum in enumerate(self.train_set):
                x, y = datum
                out = self.app.model(x.cuda())
                loss = self.app.loss_function(out, y)
                train_loss += loss.data.item()

                # Optimize the params in network
                self.app.optimizer.zero_grad()
                loss.backward()
                self.app.optimizer.step()

            train_loss /= len(self.train_set)
            # Log the train loss for tensorboard
            self.app.train_summary.add_scalar('Train_Loss', train_loss, epoch)
            print('Train finished, loss=%f' % train_loss, end='')

            # Start validating
            self.app.model.eval()
            val_loss = 0.
            # Init a R2score instance
            r2score = reg.R2Score(device='cuda')
            pred = []
            for _, datum in enumerate(self.validate_set):
                x, y = datum
                out = self.app.model(x.cuda())
                loss = self.app.loss_function(out, y)
                for i in out.tolist():
                    pred.extend(i)
                val_loss += loss.data.item()
                r2score.update((out, y))

            val_loss /= len(self.validate_set)
            print('Test finished, loss=%f, r2=%f' % (val_loss, r2score.compute()))

            # Log the validate loss & r2 for tensorboard
            self.app.train_summary.add_scalar('test_r2', r2score.compute(), epoch)
            self.app.train_summary.add_scalar('Test_Loss', val_loss, epoch)

            self.summary(epoch, y_data, pred)

    def run(self):
        self.app.network_summary(self.app.model, (1, 487, 487))
        self.create_dataset()
        self.train_network(self.app.config('training.epochs'))
