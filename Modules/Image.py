class Image:
    """ Image Class
    Image & basic attributes stored

    Attributes:
        path: The path this image stored in
        rgb: Original images with rgb color
        grayscale: Grayscale image
        percentage: the percentage of UO2 content
        thermal: Thermal Conductivity (W·m/K)
    """

    def __init__(self, path, rgb, grayscale, percentage, thermal):
        self.path = path
        self.rgb = rgb
        self.grayscale = grayscale
        self.percentage = percentage
        self.thermal = thermal
        pass

    def cuda(self):
        self.grayscale = self.grayscale.cuda()
        self.thermal = self.thermal.cuda()
        return self

    def cpu(self):
        self.grayscale = self.grayscale.cpu()
        self.thermal = self.thermal.cpu()
        return self
