from PIL import Image

import Modules


class ImageLoader:
    """ Image Loader Class
    A static tools class
    """

    @staticmethod
    def load(path, box):
        """ Load one image from given path

        Load an image form the path
        Then calculate its attributes

        Args:
            path: A path of the image to load
            box: The box of wanted image

        Return:
            image: An image instance
        """
        image = Image.open('images/RGB' + path[0] + '.png')
        gray = Image.open('images/GS' + path[0] + '.png')
        return Modules.Image(
            path[0],
            ImageLoader.pre_process(image, box),
            ImageLoader.pre_process(gray, box),
            path[1],
            path[2]
        )

    @staticmethod
    def loads(manifest):
        """ Load images from a paths list

        Arg:
            manifest: A images' paths list file

        Return:
            images: A images list
        """
        paths = []
        with open(manifest, 'r') as file:
            while True:
                line = file.readline()
                if line:
                    paths.append(tuple(line.split()))
                else:
                    break

        images = []
        for path in paths:
            image = ImageLoader.load(path, (24, 25, 1024-26, 1024-25))
            images.append(image)

        return images

    @staticmethod
    def pre_process(image, box):
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

        return new_img


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
