from ignite.contrib.metrics.regression import R2Score
from ignite.metrics.root_mean_squared_error import RootMeanSquaredError
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
        self.train_set = DataLoader(train_set, batch_size=8)
        self.validate_set = DataLoader(validate_set, batch_size=8)

    def run(self):
        self.app.network_summary(self.app.model, (1, 256, 256))
        self.create_dataset()
        y_tmp = list(map(lambda k: k[1].tolist(), self.validate_set))
        y_data = [datum for batch in y_tmp for item in batch for datum in item]

        epochs = self.app.config('training.epochs')

        def summary(epoch: int, pred: list):
            save(
                self.app.model.state_dict(),
                './storage/bin/' + self.app.helper.time_name() + '/model-' + str(epoch) + '.pth'
            )
            plot.plot(y_data, pred, '.')
            plot.plot(y_data, y_data, '-')
            plot.savefig('./storage/logs/plots/' + self.app.helper.time_name() + '/result-' + str(epoch) + '.png')
            plot.cla()

        def train_network(epoch):
            print('Epoch: %d' % epoch)
            self.app.model.train()
            train_loss = 0.
            for _, datum in enumerate(self.train_set):
                x, y = datum
                out = self.app.model(x)
                loss = self.app.loss_function(out, y)
                train_loss += loss.data.item()

                # Optimize the params in network
                self.app.optimizer.zero_grad()
                loss.backward()
                self.app.optimizer.step()

            train_loss /= len(self.train_set)

            # Log the train loss for tensorboard
            self.app.train_summary.add_scalar('Train_Loss', train_loss, epoch)
            self.app.train_summary.add_scalar('lr', self.app.lr_scheduler.get_last_lr()[0], epoch)

            if epoch > 0.8 * epochs:
                self.app.lr_scheduler.step()

            print('Train finished, loss=%f' % train_loss, end=' ')

        def validate_network(epoch):
            # Start validating
            self.app.model.eval()
            val_loss = 0.
            # Init a R2score instance
            r2score = R2Score(device='cuda')
            rmse = RootMeanSquaredError(device='cuda')
            pred = []
            for _, datum in enumerate(self.validate_set):
                x, y = datum
                out = self.app.model(x)
                loss = self.app.loss_function(out, y)
                for i in out.tolist():
                    pred.extend(i)
                val_loss += loss.data.item()
                r2score.update((out, y))
                rmse.update((out, y))

            val_loss /= len(self.validate_set)
            print('Test finished, loss=%f, r2=%f, rmse=%f' % (val_loss, r2score.compute(), rmse.compute()))

            # Log the validate loss & r2 for tensorboard
            self.app.train_summary.add_scalar('test_r2', r2score.compute(), epoch)
            self.app.train_summary.add_scalar('Test_Loss', val_loss, epoch)
            self.app.train_summary.add_scalar('test_rmse', rmse.compute(), epoch)

            summary(epoch, pred)

        def before():
            import os
            os.mkdir('./storage/logs/plots/' + self.app.helper.time_name())
            os.mkdir('./storage/bin/' + self.app.helper.time_name())

        before()
        for e in range(epochs):
            train_network(e)
            validate_network(e)

        save(
            self.app.model,
            './storage/bin/' + self.app.helper.time_name() + '/model.pth'
        )
