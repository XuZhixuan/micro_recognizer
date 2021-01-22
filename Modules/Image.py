class Image:
    """ Image Class
    Image & basic attributes stored

    Attributes:
        path: The path this image stored in
        origin: Original images with rgb color
        grayscale: Grayscale image
        size: the size of image
        thermal: Thermal Conductivity (W·m/K)
    """

    def __init__(self, path, origin, grayscale, size, thermal):
        self.path = path
        self.origin = origin
        self.grayscale = grayscale
        self.size = size
        self.thermal = thermal
        pass
