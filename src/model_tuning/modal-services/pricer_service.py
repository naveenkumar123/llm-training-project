"""
@author: Naveen N G
@date: 08-10-2025
@description: Deploy the finetuned model using Modal platform
"""



import modal
from modal import App, Image

# Login here https://modal.com, and setup the secrete token of hugging face. After that run the command : "modal setup"  to generate a modl platform tokens.
app = modal.App("pricer-service")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")

# This collects the secret from Modal.
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = "naveenng10/product-price-2025-10-07_14.46.04"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    import os
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    QUESTION = "How much does this cost to the nearest dollar?"
    PREFIX = "Price is $"

    prompt = f"{QUESTION}\n{description}\n{PREFIX}"
    
    # Quant Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        quantization_config=quant_config,
        device_map="auto"
    )

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    result = tokenizer.decode(outputs[0])

    contents = result.split("Price is $")[1]
    contents = contents.replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0


# To deploy this to production in modal platfom, follow the below steps:
# 1. Move the directory of this file 
# 2. modal deploy pricer_service