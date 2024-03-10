# Live StableDiffusion

Generate images using StableDiffusion, while looking at the intermediate states that the image goes through until it finishes the process.

## Installation

```bash
pip install .
```

## Observe intermediate images in files

Use the CLI interface for this use case:

```bash
# instructions
python3 -m live_stablediffusion -h

# example usage
python3 -m live_stablediffusion \
    -d my_directory/ \
    -o my_prompt_image \
    -m CompVis/stable-diffusion-v1-4 \
    --pipeline-opts num_inference_steps=20 \
    "Coconut tree on an island"
```
