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
        self.train_set = DataLoader(train_set)
        self.validate_set = DataLoader(validate_set)

    def summary(self):
        save(self.app.model, './storage/bin/model-' + time_name() + '.pt')
        self.app.model.eval()
        y_data = []
        pred = []
        for i, datum in enumerate(self.validate_set):
            x, y = datum
            pred.append(self.app.model(x))
            y_data.append(y)

        plot.plot(y_data, pred)
        plot.savefig('./storage/print/' + time_name() + '.png')

    def train_network(self, epochs):
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
            for _, datum in enumerate(self.validate_set):
                x, y = datum
                out = self.app.model(x.cuda())
                loss = self.app.loss_function(out, y)
                val_loss += loss.data.item()
                r2score.update((out, y))

            val_loss /= len(self.validate_set)
            print('Test finished, loss=%f, r2=%f' % (val_loss, r2score.compute()))

            # Log the validate loss & r2 for tensorboard
            self.app.train_summary.add_scalar('test_r2', r2score.compute(), epoch)
            self.app.train_summary.add_scalar('Test_Loss', val_loss, epoch)

    def run(self):
        self.app.network_summary(self.app.model, (1, 487, 487))
        self.create_dataset()
        self.train_network(self.app.config('training.epochs'))
        self.summary()
