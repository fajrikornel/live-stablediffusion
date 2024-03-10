from abc import abstractmethod
from PIL import Image
from diffusers import StableDiffusionPipeline

class AbstractImageGenerator:

    @abstractmethod
    def generate_image(self, prompt: str) -> Image:
        raise NotImplementedError("Not implemented!")


class MPSImageGenerator(AbstractImageGenerator):

    def __init__(self, model_name: str, **pipe_options) -> None:
        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()

        self.pipe = pipe
        self.pipe_options = pipe_options

    def generate_image(self, prompt: str) -> Image:
        return self.pipe(prompt, **self.pipe_options).images[0]
