import Modules
import Tools

import numpy

from helper import *


class Program:
    def __init__(self):
        self.train_set = None
        self.validate_set = None
        self.data = Tools.ImageLoader.loads('images/load_list.txt')
        self.model = Modules.Network()

    def create_dataset(self):
        """ Create the dataset for training & validation
        Convert from Image instance to dataset
        """
        gray_images = []
        scale = []
        for image in self.data:
            gray_images.append(numpy.array(image.grayscale))
            scale.append(image.thermal)

        x_train, x_validate, y_train, y_validate = train_test_split(gray_images, scale, validate_size=0.3)

        self.train_set = (x_train, y_train)
        self.validate_set = (x_validate, y_validate)

    def train_network(self):
        history = self.model.net.fit(

        )
        return history

    def main(self):
        self.create_dataset()
        self.train_network()
        pass


if __name__ == '__main__':
    app = Program()
    app.main()
