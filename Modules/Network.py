import tensorflow.keras as keras


class Network:
    """ Neural Network Class
    Neural Network definition & compilation
    With Keras API used

    Attributes:
        net: Neural Network model with Keras API
    """

    def __init__(self):
        self.net = keras.Sequential()
        self.set_network()
        self.compile_network()

    def set_network(self):
        """ Define Neural Network
        This function define the layers of an network
        """
        self.net.add(keras.layers.Conv2D(
            filters=32,
            kernel_size=(36, 36),
            activation='relu',
            input_shape=(300, 300)
        ))

        self.net.add(keras.layers.Conv2D(
            filters=32,
            kernel_size=(36, 36),
            activation='relu',
            input_shape=(300, 300)
        ))

        self.net.add(keras.layers.MaxPool2D())

    def compile_network(self):
        """ Compile Neural Network
        This function compile the network with given params
        """
        self.net.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
