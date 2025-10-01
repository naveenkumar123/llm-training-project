"""
@author: Naveen N G
@date: 1-10-2025
@description: Upload the trained and test prompt dataset to Huggingface.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from hugging_face_models.Login import login_hf
import random
import pickle

load_dotenv()
login_hf()

class UploadDataset:

    def __init__(self, items):
        self.sample = items

    def upload(self):
        random.seed(42)
        random.shuffle(self.sample)
        train = self.sample[:40_000] # It's typical to use 5%-10% of our data for testing purposes, i have taken more.
        test = self.sample[40_000:45_000] # 52,000 for testing
        print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")

        # print(train[0].prompt)
        # print(test[0].test_prompt())

        train_prompts = [item.prompt for item in train]
        train_prices = [item.price for item in train]
        test_prompts = [item.test_prompt() for item in test]
        test_prices = [item.price for item in test]


        # Create a Dataset from the lists
        train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
        test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

        DATASET_NAME = "naveenng10/product-price-data"
        dataset.push_to_hub(DATASET_NAME, private=True)

        # Let's pickle the training and test dataset so we don't have to execute all this code next time!
        with open('train.pkl', 'wb') as file:
            pickle.dump(train, file)

        with open('test.pkl', 'wb') as file:
            pickle.dump(test, file)