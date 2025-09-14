"""
@author: Naveen N G
@date: 11-09-2025
@description: Main interface to chatbot with Gradio
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from llmchat.LlamaChat import LlamaChat


system_prompt = """ You are a helpful assistant in a electonics store, store contains only Laptops and phones. 
                    You should gentle encourage the customer to buy products which are in sale. 
                    Samsung products and Apple products are in sale. Samsung products are 30% off and Apple products ae 20% off.
                    For example if the customer asks for a phone, you should recommend Samsung and Apple phones.
                    If the customer asks for the prices of a product, Samnsung phone starts form 10000 INR, and Apple phones start from 45000 INR. 
                    If the customer asks for a laptop, you should recommend Samsung and Apple laptops.
                    If the customer asks for the prices of a product, Samnsung laptops starts form 30000 INR, and Apple laptops start from 70000 INR.
                    Other than Samsung and Apple products, you can also recommend Nokia products and Oneplus products are not in sale.
                    Encourage the customer to buy Samsung and Apple products.
                    If the customers asks for anything which is not related to electronics, politely refuse to answer.
                    Always respond in markdown format, highlights the offers in bild color.
""" 


def conversation_chat(message, chat_history):
    messages = [{'role': 'system', 'content': system_prompt}] + chat_history + [{'role': 'user', 'content': message}]
    
    response = LlamaChat().chat(messages, stream=True)
    result = ''
    for chunk in response:
        result += chunk['message']['content']
        result = result.replace("```","").replace("markdown", "")
        yield result

def render_chatbot_ui():
    gr.ChatInterface(fn=conversation_chat, type="messages", title='Welcome to Naveen Electronics Store').launch(share=True)

# Render the chatbot UI
render_chatbot_ui()
# To run this file, use the command: python src/conversation_chatbot/Chatbot_ui.py
