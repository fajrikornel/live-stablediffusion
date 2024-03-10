from abc import abstractmethod
import os
from PIL import Image

class ImageSubscriber:

    @abstractmethod
    def __call__(self, image: Image) -> None:
        raise NotImplementedError("Not implemented!")


class SaveToFileImageSubscriber(ImageSubscriber):

    def __init__(self, image_name: str) -> None:
        self.image_name = image_name
        self.iteration = 0

    def __call__(self, image: Image) -> None:
        image_name = self.__format_image_name()
        dirname = os.path.dirname(image_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        image.save(image_name)
        self.iteration += 1

    def __format_image_name(self):
        return f"{self.image_name}_{self.iteration}.png"
