from PIL import Image
from torchvision import transforms

from typing import TypeVar

import Modules


loader = transforms.Compose([
    transforms.ToTensor()
])


LoadList = TypeVar('LoadList', str, list)


class ImageLoader:
    """ Image Loader Class
    A static tools class
    """

    @staticmethod
    def load(
        num: str,
        path: str,
        percentage: float,
        thermal: float,
        box: tuple
    ) -> Modules.Image:
        """ Load one image from given path

        Load an image form the path
        Then calculate its attributes

        Args:
            num: The num of image
            path: the path of the gray image to load
            percentage: UO2 percentage
            thermal: Thermal conductivity
            box: The box of wanted image

        Return:
            image: An image instance
        """
        image = Image.open('images/RGB' + num + '.png')
        gray = Image.open('images/GS' + num + '.png')
        return Modules.Image(
            path=path,
            rgb=ImageLoader.pre_process(image, box, 0.5),
            grayscale=ImageLoader.pre_process(gray, box, 0.5),
            percentage=float(percentage),
            thermal=float(thermal)
        )

    @staticmethod
    def loads(manifest: LoadList) -> list:
        """ Load images from a paths list

        Arg:
            manifest: A images' paths list file

        Return:
            images: A images list
        """
        paths = []
        if isinstance(manifest, str):
            with open(manifest, 'r') as file:
                while True:
                    line = file.readline()
                    if line:
                        paths.append(line.split())
                    else:
                        break
        else:
            for material in manifest:
                paths.append((
                    material[0],
                    material[3],
                    material[1],
                    material[2]
                ))

        images = []
        for path in paths:
            image = ImageLoader.load(*path, box=(24, 25, 1024 - 26, 1024 - 25))
            images.append(image)

        return images

    @staticmethod
    def pre_process(image: Image, box: tuple, size: float = 0.):
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

        return loader(new_img).unsqueeze(0).to('cuda')


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
