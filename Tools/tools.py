from typing import Union, Tuple, List

from PIL import Image
from torch import nn
from torch import optim
from torchvision import transforms

import Modules
from container import Container

LoadList = List[Tuple[str, str, str, float, float]]


class ImageLoader:
    """ Image Loader Class
        A tool class
    """

    def __init__(self, app: Container, size: Union[float, Tuple] = .5):
        self.size = size
        self.app = app
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize()
        ])

    def __call__(self, *args):
        return self.loads(*args)

    def load(
            self,
            origin: str,
            path: Tuple[str, str],
            percentage: float,
            thermal: float,
            box: tuple
    ) -> Modules.Image:
        """ Load one image from given path

        Load an image form the path
        Then calculate its attributes

        Args:
            origin: The origin path of image
            path: the paths of the image to load
            percentage: UO2 percentage
            thermal: Thermal conductivity
            box: The box of wanted image

        Return:
            image: An image instance
        """
        image = Image.open(path[0])
        gray = Image.open(path[1])
        return Modules.Image(
            path=origin,
            rgb=None,  # self.pre_process(image, box),  # Not loading rgb image
            grayscale=self.pre_process(gray, box),
            percentage=float(percentage),
            thermal=from_numpy(
                # Normalize the thermal conductivity
                numpy.array(
                    (image.thermal - self.app.config('data.bound.low')) / self.app.config('data.bound.inter'))
            ).view(1, 1).float().cuda()
        )

    def loads(self, manifest: LoadList) -> list:
        """ Load images from a paths list

        Arg:
            manifest: A images' paths list file

        Return:
            images: A images list
        """
        images = []
        for material in manifest:
            image = self.load(
                origin=material[0],
                path=(material[1], material[2]),
                percentage=material[3],
                thermal=material[4],
                box=(24, 25, 1024 - 26, 1024 - 25)
            )
            images.append(image)

        return images

    def pre_process(self, image: Image, box: tuple) -> Image:
        """ Process an image loaded

        Args:
            image: The origin image
            box: The box of wanted image

        Returns:
            new_image: The cropped image
        """
        shape = (
            box[2] - box[0],
            box[3] - box[1]
        )

        if image.size > shape:
            new_img = image.crop(box)
        elif image.size < shape:
            raise SizeTooSmallException(image.filename, shape, image.size)
        else:
            new_img = image

        if self.size:
            if isinstance(self.size, tuple):
                new_img = new_img.resize(self.size, Image.ANTIALIAS)
            elif isinstance(self.size, float):
                new_img = new_img.resize(
                    (int(shape[0] * self.size), int(shape[1] * self.size)), Image.ANTIALIAS)

        return self.loader(new_img).to('cuda')


class SizeTooSmallException(Exception):
    """ Size too small exception

    Attributes:
        path: the path to image
        shape_required: The shape crop to
        shape_given: The shape of given image
    """

    def __init__(self, path, shape_required, shape_given):
        """ Initializer of exception class

        Args:
            path: the path to image
            shape_required: The shape crop to
            shape_given: The shape of given image
        """
        self.path = path
        self.shape_required = shape_required
        self.shape_given = shape_given

    def __str__(self):
        """ Message

        Return: The exception message
        """
        return 'The size required is %d×%d, %d×%d given in ' % (
            self.shape_required[0], self.shape_required[1],
            self.shape_given[0], self.shape_given[1],
        ) + self.path


class ClsInitializer:
    def __init__(self):
        pass

    def __call__(self, name: str, args: dict):
        return self.init(name, args)

    @staticmethod
    def init(name: str, args: dict = None):
        """
        Find a class related to reference and instantiate it
        Args:
            name: The name of class in torch.nn
            args: The args used to instantiate

        Returns:
            An object
        """
        cls = None
        if args is None:
            args = {}

        if name in nn.modules.__all__:
            cls = getattr(nn, name)
        else:
            try:
                cls = getattr(optim, name)
            except AttributeError:
                exit('No module named %s found in torch' % name)

        return cls(**args)


class UnbufferedLogger:
    def __init__(self, app: Container):
        from helper import time_name
        self.app = app
        self.log_file = open(self.app.config(
            'logs.print.dir') + time_name() + '.txt', 'w')

    def log(self, content: str, level: str):
        self.write('[ %s ] %s' % (level, content))

    def write(self, content):
        self.log_file.write(content)
        self.log_file.flush()

    def __call__(self, content: str, level: str):
        self.log(content, level)

    def __getattr__(self, item):
        return getattr(self.log_file, item)
