import os

import torch
import numpy
import torchvision
import torchsummary
import matplotlib.pyplot as plot

from ignite.contrib.metrics.regression import R2Score

from tensorboardX import SummaryWriter

from Source import *
from Modules import *
from Tools import *

from helper import *


def debug():
    ClsInitializer.init('sgd')

    source = SavedSource('./bin/data-20210427T201647.pkl')
    data = []

    low = min(source, key=lambda k: k.thermal).thermal
    inter = max(source, key=lambda k: k.thermal).thermal - low

    for image in source:
        data.append(
            (
                image.grayscale,
                torch.from_numpy(
                    numpy.array((image.thermal - low) / inter)  # Normalize the thermal conductivity
                ).view(1, 1).float().cuda()
            )
        )

    model = torch.load('./bin/model-20210427T162628.pt')
    torchsummary.summary(model, (1, 487, 487))
    r2s = R2Score(device='cuda')
    y_data = list(map(lambda k: k[1].item(), data))
    pred = []
    for i, datum in enumerate(data):
        x, y = datum
        out = model(x)
        r2s.update((out, y))
        pred.append(out.item())

    plot.plot(y_data, y_data, '-')
    plot.plot(y_data, pred, '.')
    plot.savefig('0.png')
    score = r2s.compute()

    pass


if __name__ == '__main__':
    debug()
