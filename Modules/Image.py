from typing import Union, Optional

from torch import Tensor


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

    def __init__(
            self,
            path: str,
            rgb: Optional[Tensor],
            grayscale: Optional[Tensor],
            percentage: float,
            thermal: Union[float, Tensor]
    ):
        self.path = path
        self.rgb = rgb
        self.grayscale = grayscale
        self.percentage = percentage
        self.thermal = thermal
        pass
