

"""
@author: Naveen N G
@date: 10-09-2025
@description: GptChat provides an interface to interact with OpenAI's chat models.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display


class GptChat:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.openai = OpenAI()
        load_dotenv(override=True)
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")


    def chat(self, messages, stream: bool = False):
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        if stream:
            return response
        else:
            return response.choices[0].message.content