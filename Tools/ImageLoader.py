from PIL import Image
from torchvision import transforms

from typing import Union, Tuple, List

import Modules

LoadList = List[Tuple[str, str, str, float, float]]


class ImageLoader:
    """ Image Loader Class
        A tool class
    """

    def __init__(self):
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
            rgb=self.pre_process(image, box, 0.5),
            grayscale=self.pre_process(gray, box, 0.5),
            percentage=float(percentage),
            thermal=float(thermal)
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

    def pre_process(self, image: Image, box: tuple, size: Union[tuple, float] = .5) -> Image:
        """ Process an image loaded

        Args:
            size: Resizing image
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

        if size:
            if isinstance(size, tuple):
                new_img = new_img.resize(size, Image.ANTIALIAS)
            elif isinstance(size, float):
                new_img = new_img.resize((int(shape[0] * size), int(shape[1] * size)), Image.ANTIALIAS)

        return self.loader(new_img).unsqueeze(0).to('cuda')


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
