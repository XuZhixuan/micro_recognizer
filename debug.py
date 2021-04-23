import torch
import numpy
import torchvision

from Source import *
from Modules import *

from helper import *


def debug():
    model = Network()
    source = SavedSource('mnist.pkl')

    # ref = torchvision.datasets.MNIST(root='./tests', download=True)

    images = []
    scales = []
    for datum in source:
        tensor, scalar = datum
        images.append(tensor)
        scales.append(
            torch.from_numpy(
                numpy.array([scalar])
            ).cuda()
        )

    x_train, x_validate, y_train, y_validate = train_test_split(images, scales, validate_size=0.1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=.001,
        momentum=.9
    )

    for epoch in range(10):
        train_loss = .0
        model.train()
        for (x, y) in zip(x_train, y_train):
            optimizer.zero_grad()
            out = model.net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print('train loss: %f' % (train_loss / len(y_train)))

        model.eval()
        correct = 0
        total = len(y_validate)
        for (x, y) in zip(x_validate, y_validate):
            out = model.net(x)
            _, pred = torch.max(out.data, 1)
            correct += (pred == y)

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    debug()
