from typing import List
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

from live_stablediffusion.subscriber import ImageSubscriber

class ImageGenerator:

    def __init__(self, pipe, **pipe_options) -> None:
        self.pipe = pipe
        self.pipe_options = pipe_options
        self.image_subscribers: List[ImageSubscriber] = []

    def run_prompt(self, prompt: str) -> Image:
        image = self.pipe(prompt,
                         **self.pipe_options,
                         callback_on_step_end=self.get_publish_image_callback(),
                         callback_on_step_end_tensor_inputs=['latents']
                         ).images[0]

        for subscriber in self.image_subscribers:
            subscriber(image)

    def register_image_subscribers(self, subscriber: ImageSubscriber):
        self.image_subscribers.append(subscriber)

    def unregister_image_subscribers(self, subscriber: ImageSubscriber):
        self.image_subscribers.remove(subscriber)

    def get_publish_image_callback(self):
        def publish_image_callback(pipe, step_index, timestep, callback_kwargs):
            latents = callback_kwargs['latents']
            with torch.no_grad():
                latents = 1 / 0.18215 * latents
                image = pipe.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = pipe.numpy_to_pil(image)

                for subscriber in self.image_subscribers:
                    subscriber(image[0])

            return callback_kwargs

        return publish_image_callback


class MPSImageGenerator(ImageGenerator):

    def __init__(self, model_name: str, **pipe_options) -> None:
        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()

        super().__init__(pipe, **pipe_options)
