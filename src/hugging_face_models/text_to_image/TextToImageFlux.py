"""
@author: Naveen N G
@date: 17-09-2025
@description: This module provides functionality to run the huggingface pipline using diffusers runs FluxPipeline for the text to image.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
import torch
from diffusers import FluxPipeline


def run_text_to_image():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU.
    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flu_cat_image.png")



login_hf() # Loging into huggingface
run_text_to_image()

# python src/hugging_face_models/text_to_image/TextToImageFlux.py