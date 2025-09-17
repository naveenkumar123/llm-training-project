
"""
@author: Naveen N G
@date: 17-09-2025
@description: This module provides functionality to run the huggingface pipline using transform library, these pipline executes the tasks in the exmaples provide.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
from transformers import pipeline
import torch

# Below code is for Mac, incase of NVIDIA chip directly pass the device='cuda'
# Check if the MPS device is available
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

# Sentiment Analysis
def sentiment_analysi():
    # Load the pipeline and run it on the MPS device
    classifier_task = pipeline("sentiment-analysis", device=device)
    result = classifier_task("I'm super excited to learn LLM enineering training.")
    print(result)
    # Result would be: [{'label': 'POSITIVE', 'score': 0.9990454316139221}]


 # Named Entity Recognition
def named_entity_recognisation():
    ner_task = pipeline("ner", grouped_entities=True, device=device)
    result = ner_task("Naveen is a working IT professional from India")
    print(result)
    # Result would be  : [{'entity_group': 'PER', 'score': 0.99538463, 'word': 'Naveen', 'start': 0, 'end': 6}, {'entity_group': 'LOC', 'score': 0.9995179, 'word': 'India', 'start': 41, 'end': 46}]


# Content Summarization
def content_summary():
    content_summary = pipeline("summarization", device=device)
    content = """
                  Neerthadi (or Nirthadi) is a village in the Davangere district of Karnataka, India, known for its Ranganatha Swamy Temple, 
                  a post-Vijayanagara Empire structure that was rebuilt after being destroyed by the armies of Aurangzeb in 1696. 
                  The temple is an important historical monument and is protected by the Archaeological Survey of India    
            """
    result = content_summary(content, max_length=50, min_length=10, do_sample=False)
    print(result)
    # Result would be:  [{'summary_text': ' Neerthadi (or Nirthadi) is a village in the Davangere district of Karnataka, India . It is known for its Ranganatha Swamy Temple, a post-Vijayanagara Empire structure .'}]

# Text generation
def text_generation():
    text_generator_task = pipeline("text-generation", device=device)
    result = text_generator_task("Tell us about India in 100 words")
    print(result[0]['generated_text'])



# sentiment_analysi()
# named_entity_recognisation()
# content_summary()
text_generation()

# Use this command to run : python src/hugging_face_models/common_tasks/Common.py 