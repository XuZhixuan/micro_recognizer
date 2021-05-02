import numpy
from torch import from_numpy, cat, save
import matplotlib.pyplot as plot

import ignite.contrib.metrics.regression as reg

from helper import *
from container import Container


class Handler:
    def __init__(self, app: Container):
        self.train_set = None
        self.validate_set = None
        self.app = app

    def create_dataset(self, batch_size: int = 1):
        """
        Create the dataset for training & validation
        Convert from Image instance to dataset
        """
        # Calculate the min thermal conductivity in the set
        low = 16.0  # min(self.source, key=lambda x: x.thermal).thermal
        inter = 28.0  # max(self.source, key=lambda x: x.thermal).thermal - low

        data = []
        for image in self.app.source:
            data.append(
                (
                    image.grayscale,
                    from_numpy(
                        numpy.array((image.thermal - low) / inter)  # Normalize the thermal conductivity
                    ).view(1, 1).float().cuda()
                )
            )

        if batch_size > 1:
            temp = []
            for i in range(0, len(data), batch_size):
                x_tensor = cat(list(map(lambda x: x[0], data[i:i + batch_size])))
                y_tensor = cat(list(map(lambda x: x[1], data[i:i + batch_size])))
                temp.append(
                    (x_tensor, y_tensor)
                )
            data = temp

        self.train_set, self.validate_set = train_test_split(data, validate_size=0.2)

    def summary(self):
        self.app.model.eval()
        y_data = map(lambda k: k[1], self.validate_set)
        pred = []
        for i, datum in enumerate(self.validate_set):
            x, y = datum
            pred.append(self.app.model(x))

        fig = plot.plot(y_data, pred)
        pass

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

            train_loss /= len(self.train_set[0])
            # Log the train loss for tensorboard
            self.app.writer.add_scalar('Train_Loss', train_loss, epoch)
            print('Train finished, loss=%f' % train_loss, end=' ')

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

            val_loss /= len(self.validate_set[0])
            print('Test finished, loss=%f, r2=%f' % (val_loss, r2score.compute()))

            # Log the validate loss & r2 for tensorboard
            self.app.writer.add_scalar('test_r2', r2score.compute(), epoch)
            self.app.writer.add_scalar('Test_Loss', val_loss, epoch)

        save(self.app.model, './bin/model-' + time_name() + '.pt')

    def run(self):
        self.create_dataset()
        self.train_network(100)
