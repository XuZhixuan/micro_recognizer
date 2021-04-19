import Modules
import Tools

import numpy
import torch

import ignite.contrib.metrics.regression as reg

from tensorboardX import SummaryWriter

from helper import *

from Source import *


class Program:
    def __init__(self):
        self.train_set = None
        self.validate_set = None

        self.writer = SummaryWriter('logs/1.log')

        # self.data =  DBSource(
        #     db_hostname='49.208.46.17',
        #     db_password='123456',
        #     db_username='root',
        #     db_name='mtrl_info_fg',
        #     base_url='http://49.208.46.17:3000'
        # )

        self.data = SavedSource('data.pkl')
        self.model = Modules.Network()
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.001)

    def create_dataset(self):
        """
        Create the dataset for training & validation
        Convert from Image instance to dataset
        """
        gray_images = []
        scale = []
        for image in self.data:
            gray_images.append(image.grayscale)
            scale.append(
                torch.Tensor(numpy.array(image.thermal)).view(1, 1).cuda()
            )

        x_train, x_validate, y_train, y_validate = train_test_split(gray_images, scale, validate_size=0.3)

        self.train_set = (x_train, y_train)
        self.validate_set = (x_validate, y_validate)

    def train_network(self, epochs):
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)
            self.model.train()
            train_loss = 0.
            for (x, y) in zip(self.train_set[0], self.train_set[1]):
                out = self.model.net(x)
                loss = self.loss_function(out, y)
                train_loss += loss.data.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Train finished, loss=%f' % (train_loss / len(self.train_set[0])), end=' ')

            self.model.eval()
            val_loss = 0.
            r2score = reg.R2Score(device='cuda')
            for (x, y) in zip(self.validate_set[0], self.validate_set[1]):
                out = self.model.net(x)
                loss = self.loss_function(out, y)
                val_loss += loss.data.item()
                r2score.update((out, y))

            print('Test finished, loss=%f, r2=%f' % ((val_loss / len(self.validate_set[0])), r2score.compute()))

        torch.save(self.model, './0.pl')

    def main(self):
        self.create_dataset()
        self.train_network(5)
        pass


if __name__ == '__main__':
    app = Program()
    app.main()
