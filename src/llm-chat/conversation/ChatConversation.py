import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display, update_display
import ollama

load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')

MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'
GEMINI_MODEL = 'gemini-2.5-flash'

openai = OpenAI()

conversation = ''

def systemPromptMsg(name):
    return f"Conversation with other 3 persons conversations is about the travel. Continue the conversation based on the other personse response."
     

def userPromptMsg(name):
    return f"""You are {name}, in conversation with other 2 persons. 
                The conversation so far is as follows:{conversation}.
                Now with this, response with what you like to say next, as {name}
             """
    
def callGpt(name):
    # systemContent = systemPromptMsg(name)
    # userContent = userPromptMsg(name)
    # print(systemContent)
    # print(userContent)
    gptResponse = openai.chat.completions.create(
        model=MODEL_GPT,
        messages = [
            {'role' : 'system' , 'content' : systemPromptMsg(name)},
            {'role' : 'user' , 'content' : userPromptMsg(name)}
        ]
    )
    return gptResponse.choices[0].message.content

def callOllama(name):
    ollamResponse = ollama.chat(
        model=MODEL_LLAMA,
        messages = [
            {'role' : 'system' , 'content' : systemPromptMsg(name)},
            {'role' : 'user' , 'content' : userPromptMsg(name)}
        ]
    )
    return ollamResponse.message.content

def callGemini(name):
    gemini_via_openai_client = OpenAI(
        api_key=google_api_key, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    geminiReponse = gemini_via_openai_client.chat.completions.create(
        model=GEMINI_MODEL,
        messages = [
            {'role' : 'system' , 'content' : systemPromptMsg(name)},
            {'role' : 'user' , 'content' : userPromptMsg(name)}
        ]
    )
    return geminiReponse.choices[0].message.content


# Start the conversation...
conversation='Hi'

for i in range(3):
    gptConversation = callGpt('Naveen')
    display(Markdown(f"**Naveen:**\n{gptConversation}\n"))
    conversation += (f"\n{gptConversation}")

    ollamaConversation = callOllama('Basu')
    display(Markdown(f"**Basu:**\n{ollamaConversation}\n"))
    conversation += (f"\n{ollamaConversation}")

    geminiConversation = callGemini('Thippa')
    display(Markdown(f"**Thippa:**\n{geminiConversation}\n"))
    conversation += (f"\n{geminiConversation}")

    
    

