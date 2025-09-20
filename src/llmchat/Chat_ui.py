
"""
@author: Naveen N G
@date: 11-09-2025

Chat_ui.py

This module provides Gradio-based user interfaces for interacting with different language models (GPT and Llama).
It defines functions to render basic and advanced chat UIs, and to handle chat interactions with selected models.

Functions:
    basic_chat_ui_render(callFunc):
        Renders a simple Gradio interface for chatting with a single model using a textbox input and output.

    chat(user_prompt_input):
        Handles a chat interaction with the GPT model using the provided user prompt.
        Returns the model's response.

    advanced_chat_ui_render(callFunc):
        Renders an advanced Gradio interface allowing users to select between GPT and Llama models,
        enter a message, and view the response in markdown format.

    multi_model_chat(user_prompt_input, model_type):
        Handles chat interactions with either the GPT or Llama model based on user selection.
        Streams and yields the response incrementally for real-time display.

Usage:
    Run this file to launch the Gradio chat UI. The advanced interface allows model selection and markdown responses.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from dotenv import load_dotenv
from GptChat import GptChat
from LlamaChat import LlamaChat



load_dotenv()

def basic_chat_ui_render(callFunc):
    gr.Interface(fn=callFunc, inputs="textbox", outputs="textbox").launch()


def chat(user_prompt_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt_input}
    ]
    return GptChat().chat(messages)


def advanced_chat_ui_render(callFunc):
    view =  gr.Interface(
                fn=callFunc,
                inputs=[gr.Textbox(label="Your message:"), gr.Dropdown(["GPT", "Llama"], label="Select model")], 
                outputs=[gr.Markdown(label="Response:")],
                flagging_mode="never")
    view.launch(share=True)



def multi_model_chat(user_prompt_input, model_type):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt_input}
    ]
    result = ''
    if model_type == "GPT":
        result = ''
        response = GptChat('gpt-4.1-nano').chat(messages, True)
        for chunk in response:
            result += chunk.choices[0].delta.content or ""
            yield result
    elif model_type == "Llama":
        result = ''
        response =  LlamaChat().chat(messages, True)
        for chunk in response:
            result += chunk.message.content or ""
            yield result


# To run this file, use the command: python src/llmchat/Chat_ui.py
# Basic interface to chat with GPT model, using Gradio. User can enter text and see the response.

# basic_chat_ui_render(chat)

# More advanced interface to chat with multiple models, using Gradio. User can enter text, select model and see the response in markdown format.
advanced_chat_ui_render(multi_model_chat)



