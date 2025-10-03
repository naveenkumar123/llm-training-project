
"""
@author: Naveen N G
@date: 03-10-2025
@description: Fine tuning llama 3.1 model for product price prediction.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from hugging_face_models.Login import login_hf
from product_price_estimate.load_data import DataLoader
import re
import math
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import matplotlib.pyplot as plt
from test_util import TestUtil

load_dotenv()
login_hf()




BASE_MODEL = 'meta-llama/Meta-Llama-3.1-8B'
DATASET_NAME = 'naveenng10/product-price-data'
QUANT_4_BIT = False

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


class LlamaFineTuning:

    def load_dataset(self):
        # Load the dataset from huggingface, earlier uploadded dataset form the upload_dataset.py 
        # dataset = load_dataset(DATASET_NAME)
        # train = dataset['train']
        # test = dataset['test']

        # Loading the dataset form local, which was saved in pickle files.
        train, test = DataLoader().get_saved_data()
        return [train, test]


    def extract_price(s):
        if "Price is $" in s:
            contents = s.split("Price is $")[1]
            contents = contents.replace(',','').replace('$','')
            match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
            return float(match.group()) if match else 0
        return 0


    def get_quantisation_config(self):
        if QUANT_4_BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        return quant_config


    def load_model(self):
        quant_config = self.get_quantisation_config()
        tokenizer.pad_token = tokenizer.eos_token # Truncating the befor and afete tokens.
        tokenizer.padding_side = "right"
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map=device,
        )
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:.1f} GB")
        self.base_model = base_model
        

    def model_predict(self, prompt):
        base_model = self.base_model
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(inputs.shape, device=device)
        outputs = base_model.generate(inputs, max_new_tokens=4, attention_mask=attention_mask, num_return_sequences=1)
        response = tokenizer.decode(outputs[0])
        return self.extract_price(response)
    

def run_test():
    llamaFineTuning = LlamaFineTuning()
    train, test = llamaFineTuning.load_dataset()
    llamaFineTuning.load_model()
    print(test[0])
    print(test[0].price)    
    # Make nessary changes incase of dataset is loaded form the huggingface.
    response = llamaFineTuning.model_predict(test[0].test_prompt())
    print(response)

    TestUtil.test(llamaFineTuning.model_predict, test)


if __name__ == "__main__":
    run_test()

# Note: run this in Google colab for quicker run.  Mac doesn't support the BitsAndBytesConfig for the 4bit and 8bit
# Use this command to run : python src/model_tuning/llama_finetunning.py