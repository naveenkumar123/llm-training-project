"""
@author: Naveen N G
@date: 17-09-2025
@description: This module provides functionality to run the huggingface AutoTokenizer using transform library, below examples show how token works.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
from transformers import AutoTokenizer  
import torch


# Make sure you have submitted the form in the huggingface for the model 'meta-llama/Meta-Llama-3.1-8B' to use, otherwise you get the access deined
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)


def text_encode_decode():    
    text = "Learn how the Tokenizers works"

    # Encoding the text
    tokens = tokenizer.encode(text)
    print(tokens)
    # Result would be : [128000, 24762, 1268, 279, 9857, 12509, 4375]
    
    # Note: Ideally, each token holds of 4 characters. lets check below.
    print(len(text)/len(tokens))

    # Decoding the text using tokens
    decoded_text = tokenizer.decode(tokens)
    print(decoded_text)
    # Result would be : <|begin_of_text|>Learn how the Tokenizers works
    # Note: begin of the decode text contains : <|begin_of_text|>, different models follow different approaches.

    # Batch decoding
    result = tokenizer.batch_decode(tokens)
    print(result)
    # Result would be : ['<|begin_of_text|>', 'Learn', ' how', ' the', ' Token', 'izers', ' works']


text_encode_decode()



# Use this command to run : python src/hugging_face_models/tokenizer/TextAutotokenizer.py 