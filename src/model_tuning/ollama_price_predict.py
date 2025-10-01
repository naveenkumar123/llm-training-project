
"""
@author: Naveen N G
@date: 1-10-2025
@description: Ollama llama model price predictor for the test data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
import re
from hugging_face_models.Login import login_hf
from llmchat.LlamaChat import LlamaChat
from product_price_estimate.load_data import DataLoader
import matplotlib.pyplot as plt
from test_util import TestUtil
import random
import math



class OllamaPricePredictor:

    def __init__(self):
        pass

    def get_price(self, s):
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0


    def messages(self, item):
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]


    def predict_price(self, item):
        messages = self.messages(item)
        response = LlamaChat().chat(messages)
        return self.get_price(response)


def run_test():
    train, test = DataLoader().get_saved_data()
    llamaPricePredictor = OllamaPricePredictor()
    print(test[0])
    TestUtil.test(llamaPricePredictor.predict_price, test)

if __name__ == "__main__":
    run_test()
    

# Use this command to run : python src/model_tuning/ollama_price_predict.py 