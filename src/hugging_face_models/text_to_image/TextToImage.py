
"""
@author: Naveen N G
@date: 17-09-2025
@description: This module provides functionality to run the huggingface pipline using transform library, these pipline executes text to image module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf

from diffusers import AutoPipelineForText2Image
import torch
from IPython.display import display
from PIL import Image

def run_text_to_image():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")

    prompt = "Fighter jets are flying in the the blue sky"
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    display(image)



login_hf() # Loging into huggingface
run_text_to_image()

# python src/hugging_face_models/text_to_image/TextToImage.py