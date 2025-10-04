
"""
@author: Naveen N G
@date: 04-10-2025
@description: Fine tunning llama model with QLoRA and hyper parameters for product price predictor.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from hugging_face_models.Login import login_hf
import re
import math
from tqdm import tqdm
from google.colab import userdata
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig
from datasets import load_dataset
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datetime import datetime
import matplotlib.pyplot as plt


load_dotenv()
login_hf()


BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "product-price"
HF_USER = "naveenng10" 
DATASET_NAME = 'naveenng10/product-price-data'
MAX_SEQUENCE_LENGTH = 182

# Run name for saving the model in the hub
RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"


# Hyperparameters for QLoRA Fine-Tuning
LORA_R = 16 # increase in 'r' leading to greater expressiveness and increase the number of trainable parameters and computational cost. 
LORA_ALPHA = 32 # alpha = 2 * r
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1
QUANT_4_BIT = True

# Hyperparameters for Training
EPOCHS = 1 # you can do more epochs if you wish, but only 1 is needed - more is probably overkill
BATCH_SIZE = 4 # on an A100 box this can go up to 16
GRADIENT_ACCUMULATION_STEPS = 1 # improve memory efficiency and stabilize training in neural networks by accumulating gradients over multiple batches before updating the model parameters
LEARNING_RATE = 1e-4 # How much the small LoRA adaptation layers are adjusted during each training step
LR_SCHEDULER_TYPE = 'cosine' # Different types: linear, cosine, constant, cosine_with_restarts, polynomial
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"


# Admin config - note that SAVE_STEPS is how often it will upload to the hub
STEPS = 50
SAVE_STEPS = 2000
LOG_TO_WANDB = True

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

class QloraPricePredictor:
    
    def __init__(self):
        pass


    # Log in to Weights & Biases
    def login_wb(self):
        wandb.login()
        # Configure Weights & Biases to record against our project
        os.environ["WANDB_PROJECT"] = PROJECT_NAME
        os.environ["WANDB_LOG_MODEL"] = "checkpoint" if LOG_TO_WANDB else "end"
        os.environ["WANDB_WATCH"] = "gradients"

        if LOG_TO_WANDB:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    def load_hf_dataset(self):
        # Load the dataset from huggingface, earlier uploadded dataset form the upload_dataset.py 
        dataset = load_dataset(DATASET_NAME)
        train = dataset['train']
        test = dataset['test']
        # Reducing the training dataset to 20,000 points
        self.train = train.select(range(20000))

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

    def set_data_collector(self):
        #  Teach the model to predict the token(s) after "Price is $" not to train the train the model to predict the description.
        response_template = "Price is $"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        self.collator = collator
    
    def lora_config(self):
        # Configuration parameters for LoRA
        lora_parameters = LoraConfig(
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            r=LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )   
        return lora_parameters
    
    def training_parameters(self):
        # General configuration parameters for training
        train_parameters = SFTConfig(
            output_dir=PROJECT_RUN_NAME,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            eval_strategy="no",
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            optim=OPTIMIZER,
            save_steps=SAVE_STEPS,
            save_total_limit=10,
            logging_steps=STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=WARMUP_RATIO,
            group_by_length=True,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            report_to="wandb" if LOG_TO_WANDB else None,
            run_name=RUN_NAME,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            dataset_text_field="text",
            save_strategy="steps",
            hub_strategy="every_save",
            push_to_hub=True,
            hub_model_id=HUB_MODEL_NAME,
            hub_private_repo=True
        )
        return train_parameters
    
    def train_model(self):

        lora_parameters = self.lora_config()
        train_parameters = self.training_parameters()
        
        # Supervised Fine Tuning Trainer(SFTT) will carry out the fine-tuning
        fine_tuning = SFTTrainer(
            model=self.base_model,
            train_dataset=self.train,
            peft_config=lora_parameters,
            args=train_parameters,
            data_collator=self.collator
        )
        # Start training
        fine_tuning.train()

        # Push our fine-tuned model to Hugging Face
        fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
        print(f"Saved to the hub: {PROJECT_RUN_NAME}")

        if LOG_TO_WANDB:
            wandb.finish()


    def start_training(self):
        self.login_wb()
        self.load_hf_dataset()
        self.load_model()
        self.set_data_collector()

        # Here we start the model training...
        self.train_model()


if __name__ == "__main__":
    QloraPricePredictor().start_training()
