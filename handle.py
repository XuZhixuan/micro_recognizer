from torch import save, max, softmax, eq
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

        def summary(epoch: int):
            save(
                self.app.model.state_dict(),
                './storage/bin/' + self.app.helper.time_name() + '/model-' + str(epoch) + '.pth'
            )

        def train_network(epoch):
            print('Epoch: %d' % epoch)
            self.app.model.train()
            train_loss = 0.
            train_correct = 0
            for _, datum in enumerate(self.train_set):
                x, y = datum
                out = self.app.model(x)
                out = softmax(out, 1)
                loss = self.app.loss_function(out, y.squeeze())
                _, out = max(out, 1)
                train_loss += loss.data.item()
                train_correct += eq(y, out).sum().item()

                # Optimize the params in network
                self.app.optimizer.zero_grad()
                loss.backward()
                self.app.optimizer.step()

            train_loss /= len(self.train_set)
            train_acc = train_correct / len(self.train_set)
            # Log the train loss for tensorboard
            self.app.train_summary.add_scalar('Train_Loss', train_loss, epoch)
            self.app.train_summary.add_scalar('Train_acc', train_acc, epoch)
            self.app.train_summary.add_scalar('LearningRate', self.app.lr_scheduler.get_last_lr()[0], epoch)
            print('Train finished, loss=%f, acc=%f' % (train_loss, train_acc), end=' ')

            self.app.lr_scheduler.step()

        def validate_network(epoch):
            # Start validating
            self.app.model.eval()
            val_loss = 0.
            val_correct = 0
            for _, datum in enumerate(self.validate_set):
                x, y = datum
                out = self.app.model(x)
                out = softmax(out, 1)
                loss = self.app.loss_function(out, y.squeeze())
                _, out = max(out, 1)
                val_loss += loss.data.item()
                val_correct += eq(out, y).sum().item()

            val_loss /= len(self.validate_set)
            val_acc = val_correct / len(self.validate_set)
            print('Test finished, loss=%f, acc=%f' % (val_loss, val_acc))

            # Log the validate loss & r2 for tensorboard
            self.app.train_summary.add_scalar('Test_acc', val_acc, epoch)
            self.app.train_summary.add_scalar('Test_Loss', val_loss, epoch)

            summary(epoch)

        def before():
            import os
            os.mkdir('./storage/logs/plots/' + self.app.helper.time_name())
            os.mkdir('./storage/bin/' + self.app.helper.time_name())

        before()
        for e in range(self.app.config('training.epochs')):
            train_network(e)
            validate_network(e)

        save(
            self.app.model,
            './storage/bin/' + self.app.helper.time_name() + '/model.pth'
        )
