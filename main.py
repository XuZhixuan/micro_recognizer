import Modules
import Tools

import numpy
import torch

from helper import *


class Program:
    def __init__(self):
        self.train_set = None
        self.validate_set = None
        self.data = Tools.ImageLoader.loads('images/load_list.txt')
        self.model = Modules.Network()
        self.loss_function = torch.nn.CrossEntropyLoss

    def create_dataset(self):
        """ Create the dataset for training & validation
        Convert from Image instance to dataset
        """
        gray_images = []
        scale = []
        for image in self.data:
            tensor = torch.from_numpy(numpy.array(image.grayscale))
            gray_images.append(tensor.cuda())
            scale.append(image.thermal)

        x_train, x_validate, y_train, y_validate = train_test_split(gray_images, scale, validate_size=0.3)

        self.train_set = (x_train, y_train)
        self.validate_set = (x_validate, y_validate)

    def train_network(self, epochs):
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)
            train_loss = 0.
            train_acc = 0.
            for x, y in self.train_set[0], self.train_set[1]:
                out = self.model.net(x)
                loss = self.loss_function(out, y)
                train_loss += loss

    def main(self):
        self.create_dataset()
        self.train_network(5)
        pass


if __name__ == '__main__':
    app = Program()
    app.main()
