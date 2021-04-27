import Modules

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

        self.writer = SummaryWriter('./logs/')

        # self.data =  DBSource(
        #     db_hostname='49.208.46.17',
        #     db_password='123456',
        #     db_username='root',
        #     db_name='mtrl_info_fg',
        #     base_url='http://49.208.46.17:3000'
        # )

        self.source = FileSource('./zips')
        self.source.dump('./bin/data-' + time_name() + '.pkl')

        config = load_json('vgg8.json')

        self.model = Modules.Network(config)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.001)

    def create_dataset(self, batch_size: int = 1):
        """
        Create the dataset for training & validation
        Convert from Image instance to dataset
        """
        # Calculate the min thermal conductivity in the set
        low = min(self.source, key=lambda x: x.thermal).thermal
        inter = max(self.source, key=lambda x: x.thermal).thermal - low

        data = []
        for image in self.source:
            data.append(
                (
                    image.grayscale,
                    torch.from_numpy(
                        numpy.array((image.thermal - low) / inter)  # Normalize the thermal conductivity
                    ).view(1, 1).float().cuda()
                )
            )

        if batch_size > 1:
            temp = []
            for i in range(0, len(data), batch_size):
                x_tensor = torch.cat(list(map(lambda x: x[0], data[i:i+batch_size])))
                y_tensor = torch.cat(list(map(lambda x: x[1], data[i:i+batch_size])))
                temp.append(
                    (x_tensor, y_tensor)
                )
            data = temp

        self.train_set, self.validate_set = train_test_split(data, validate_size=0.2)

    def train_network(self, epochs):
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)
            self.model.train()
            train_loss = 0.
            for i, datum in enumerate(self.train_set):
                x, y = datum
                out = self.model(x.cuda())
                loss = self.loss_function(out, y)
                train_loss += loss.data.item()

                # Optimize the params in network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss /= len(self.train_set[0])
            # Log the train loss for tensorboard
            self.writer.add_scalar('Train_Loss', train_loss, epoch)
            print('Train finished, loss=%f' % train_loss, end=' ')

            # Start validating
            self.model.eval()
            val_loss = 0.
            # Init a R2score instance
            r2score = reg.R2Score(device='cuda')
            for i, datum in enumerate(self.validate_set):
                x, y = datum
                out = self.model(x.cuda())
                loss = self.loss_function(out, y)
                val_loss += loss.data.item()
                r2score.update((out, y))

            val_loss /= len(self.validate_set[0])
            print('Test finished, loss=%f, r2=%f' % (val_loss, r2score.compute()))

            # Log the validate loss & r2 for tensorboard
            self.writer.add_scalar('test_r2', r2score.compute(), epoch)
            self.writer.add_scalar('Test_Loss', val_loss, epoch)

        torch.save(self.model, './bin/model-' + time_name() + '.pt')

    def main(self):
        self.create_dataset()
        self.train_network(100)


def run():
    check_dir()
    app = Program()
    app.main()


if __name__ == '__main__':
    run()
