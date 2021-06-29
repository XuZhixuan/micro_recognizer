import numpy
from cv2 import imwrite, cvtColor, COLOR_GRAY2RGB
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from app import Application


def visualize():
    app = Application()
    cam = AblationCAM(
        app.model,
        app.model.layer4[-1]
    )

    source = [(i[0].cpu(), i[1].cpu()) for i in app.source if i[1].item() >= 0.2610552]

    for i, datum in enumerate(source[0:10]):
        img, _ = datum
        print(_ * 25 + 15)
        grayscale = cam(
            input_tensor=img.unsqueeze(0),
            # aug_smooth=True
        )
        grayscale = grayscale[0, :]

        rgb_img = numpy.array(img.cpu()).squeeze(0)
        rgb_img = cvtColor(rgb_img, COLOR_GRAY2RGB)

        visualization = show_cam_on_image(rgb_img, grayscale)
        imwrite(str(i) + '_cam.jpg', visualization)


if __name__ == '__main__':
    visualize()
