"""
@author: Naveen N G
@date: 18-09-2025
@description: This module provides functionality to run the huggingface BitsAndBytesConfig, AutoModelForCausalLM using transform library fro Quantization.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc

login_hf() # Loging into huggingface

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

def quantifierConfig():
    # Quantization Config - this allows us to load the model into memory and use less memory
    quantifier_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    return quantifier_config


def generate_text(model, messages):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quantifierConfig()) #  it downloads the model weights and configuration to local temp memeory 
    outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
    del model, inputs, tokenizer, outputs, streamer
    gc.collect()
    torch[device].empty_cache() # Clearning the model from the temp memory.

messages = [
    {"role": "system", "content": "You are a helpful sports assistant"},
    {"role": "user", "content": "What are all the famous sports in India"}
  ]

llama_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# phi3_model = "microsoft/Phi-3-mini-4k-instruct"
generate_text(llama_model, messages)

# Use this command to run : python src/hugging_face_models/tokenizer/Quantifier.py 