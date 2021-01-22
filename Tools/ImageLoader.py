from PIL import Image

import Modules


class ImageLoader:
    """ Image Loader Class
    A static tools class
    """

    @staticmethod
    def load(path, shape):
        """ Load one image from given path

        Load an image form the path
        Then calculate its attributes

        Args:
            path: A path of the image to load
            shape: The shape of wanted image

        Return:
            image: An image instance
        """
        image = Image.open('images/' + path[0])
        image, gray = ImageLoader.pre_process(image, shape)
        return Modules.Image(path[0], image, gray, shape, path[1])

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
            image = ImageLoader.load(path, (300, 300))
            images.append(image)

        return images

    @staticmethod
    def pre_process(image, shape):
        """ Process an image loaded

        Args:
            image: The origin image
            shape: The shape of wanted image

        Returns:
            new_image: The cropped image
            gray_image: The gray cropped image
        """
        if image.size > shape:
            box_height = image.size[0] - shape[0]
            box_height = int(box_height/2)

            box_width = image.size[1] - shape[1]
            box_width = int(box_width/2)

            box = (
                box_width,
                box_height,
                box_width + shape[1],
                box_height + shape[0]
            )
            new_img = image.crop(box)
        elif image.size < shape:
            raise SizeTooSmallException(image.filename, shape, image.size)
        else:
            new_img = image

        gray_image = new_img.convert('L')
        return new_img, gray_image


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
        return 'The size required is %d×%d, %d×%d given in '%(
            self.shape_required[0], self.shape_required[1],
            self.shape_given[0], self.shape_given[1],
        ) + self.path
