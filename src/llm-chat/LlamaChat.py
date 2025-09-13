

import os
"""
@author: Naveen N G
@date: 2024-10-03
@description: LlamaChat provides a simple interface for interacting with an Ollama-powered LLM chat model.
"""
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
import ollama

class LlamaChat:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        load_dotenv(override=True)

    def chat(self, messages):
        response = ollama.chat(
            model=self.model,
            messages=messages,
        )
        return response['message'].content