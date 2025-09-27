"""
@author: Naveen N G
@date: 27-09-2025
@description: Mobile shop application using RAG technique.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
from llmchat.GptChat import GptChat
from llmchat.LlamaChat import LlamaChat
from IPython.display import Markdown, display
import gradio as gr
import glob
import csv


current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

context = {}
def get_companies():
    data_folder_path = os.path.join(current_dir, "data", "company")
    companies = glob.glob(f"{data_folder_path}/*.md")
    comapny_name = ''
    for company in companies:
        name = company.split('/')[-1].split('.')[0]
        details = ''
        with open(company, "r", encoding="utf-8") as f:
            details = f.read()
        context[name+' company info'] = details
        comapny_name += name + ','
    
    context['brands'] = comapny_name[:-1]
    context['companies'] = comapny_name[:-1]

def get_mobiles_models():
    data_folder_path = os.path.join(current_dir, "data", "mobile-price")
    mobile_models = glob.glob(f"{data_folder_path}/*.csv")
    for mobile_model in mobile_models:
        model_filename= mobile_model.split('/')[-1]
        with open(mobile_model, "r", encoding="utf-8") as csvFile:
            csv_reader = csv.DictReader(csvFile)
            company_name = model_filename.split('.')[0].split('_')[0]
            # model_price_details = f'Below is the {company_name} models, prices details and model configurations \n\n'
            models = ''
            for row in csv_reader:
                index = 0
                modelName = ''
                model_price_details = ''
                for item in row.items():
                    key = item[0]
                    value = item[1]
                    if(index == 0):
                         models += value + ','
                         modelName += value
                    else:
                        model_price_details += f"{key}:{value},"
                    index += 1
                context[modelName + "price"] = model_price_details[:-1]
            context[company_name + ' models'] = models[:-1]

def get_assistants_context(message):
    relevant_context = []

    for context_title, context_details in context.items():
        if context_title.lower() in message.lower() or context_details.lower() in message.lower():
            print('Inside relevant_context')
            relevant_context.append(context_details)
    return relevant_context    

def add_context(user_message):
    relevant_context = get_assistants_context(user_message)
    if relevant_context:
        user_message += "\n\nThe following context must be used for this question, response based on the provided context:\n\n"
        for relevant in relevant_context:
            user_message += relevant + "\n\n"
    return user_message


def mobile_shop_assistant(message, history):
    system_message = """You are a helful assistant for Naveen mobile shop it sles only mobiles, 
                        You are provided with the following information, you must answering accurate questions:
                        - List of mobile companies 
                        - List of mobile brands,
                        - Price details of each mobile brand
                        - Mobile brand includes, Model, Color, Memory, Storage, Rating, Selling Price, Original Price
                        Give a co, accurate answers. If you don't know the answer, say so.\n
                        Do not make anything up, always be accurate.\n
                        Do not make anything up if you haven't been provided with relevant context
                        Repond only based on the context provided, do not suggest anything before the context.
                    """
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message)
    print(message)
    messages.append({"role": "user", "content": message})

    streamResponse = LlamaChat().chat(messages, True)
    response = ''
    for chunk in streamResponse:
        response += chunk.message.content or ''
        response = response.replace("```","").replace("markdown", "")
        yield response


def render_mobile_shop_ui():
    gr.ChatInterface(
        fn=mobile_shop_assistant, 
        type="messages", 
        title="Welcome to Naveen's Mobile Shop").launch(share=True)

get_companies()
get_mobiles_models()
render_mobile_shop_ui()

# Use this command to run : python src/rag/mobile-shop/shop.py 
