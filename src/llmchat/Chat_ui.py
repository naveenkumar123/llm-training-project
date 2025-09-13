import os
import gradio as gr
from GptChat import GptChat
from LlamaChat import LlamaChat


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



