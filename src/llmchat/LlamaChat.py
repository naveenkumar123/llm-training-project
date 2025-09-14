
"""
@author: Naveen N G
@date: 10-09-2025
@description: LlamaChat provides a simple interface for interacting with an Ollama-powered LLM chat model.
"""

import os
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
import ollama

class LlamaChat:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        load_dotenv(override=True)

    def chat(self, messages, stream: bool = False, response_format: str = ''):
        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            format=response_format
        )
        if stream:
            return response
        else:
            return response['message'].content