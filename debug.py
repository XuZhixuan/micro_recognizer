import os

import torch
import numpy
import torchvision

from tensorboardX import SummaryWriter

from Source import *
from Modules import *

from helper import *


def debug():
    source = FileSource('./zips')
    images = []
    for i, datum in enumerate(source):
        images.append(
            datum.grayscale
        )
    writer = SummaryWriter('./logs')
    writer.add_images(
        'train_images',
        torch.cat(images)
    )


if __name__ == '__main__':
    debug()
