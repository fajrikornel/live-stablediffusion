from argparse import ArgumentParser, Namespace
import json
import os

from live_stablediffusion.image_generator import ImageGenerator, MPSImageGenerator
from live_stablediffusion.subscriber import SaveToFileImageSubscriber

class SaveToFileCommandLineInterface:

    PROGRAM_NAME = 'Live StableDiffusion - save to file'
    PROGRAM_DESCRIPTION = 'Inputs a prompt, saves the intermediate and final images to files'
    PROGRAM_EPILOG = 'Currently only supports MPS stable diffusion'

    def __init__(self) -> None:
        parser = ArgumentParser(prog=self.PROGRAM_NAME, description=self.PROGRAM_DESCRIPTION)
        parser.add_argument('prompt', help='Image prompt', type=str)
        parser.add_argument('-o', '--output-filename', help='Output filename', type=str, required=True)
        parser.add_argument('-d', '--output-directory', help='Output directory', type=str)
        parser.add_argument('-m','--model-name', help='Model name (default: CompVis/stable-diffusion-v1-4)', type=str, default='CompVis/stable-diffusion-v1-4')
        parser.add_argument('--pipeline-opts', help='Comma-separated additional pipeline options (default: num_inference_steps=20)', type=str, default='num_inference_steps=20')

        self.parser = parser

    def start_program(self):
        args = self.parser.parse_args()

        image_generator = self.build_image_generator(args)
        subscriber = self.build_subscriber(args)

        image_generator.register_image_subscribers(subscriber)

        prompt = args.prompt

        image_generator.run_prompt(prompt)

    def build_image_generator(self, args: Namespace) -> ImageGenerator:
        model_name = args.model_name
        pipeline_opts = {}
        for pipeline_opt_pair in args.pipeline_opts.split(','):
            pipeline_opt_pair_split = pipeline_opt_pair.split('=')
            pipeline_opt = pipeline_opt_pair_split[0]
            pipeline_opt_value = json.loads(pipeline_opt_pair_split[1])

            pipeline_opts[pipeline_opt] = pipeline_opt_value

        return MPSImageGenerator(model_name=model_name, **pipeline_opts)

    def build_subscriber(self, args: Namespace) -> SaveToFileImageSubscriber:
        if args.output_directory is not None:
            output_path = os.path.join(args.output_directory, args.output_filename)
        else:
            output_path = args.output_filename

        return SaveToFileImageSubscriber(output_path)
