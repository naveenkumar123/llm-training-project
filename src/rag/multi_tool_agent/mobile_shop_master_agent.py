"""
@author: Naveen N G
@date: 29-09-2025
@description: Mobile shop application using RAG technique with langchain multi tools agents.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import gradio as gr
from csv_tool_agent import csv_tool_agent
from text_tool_agent import text_tool_agent
from llms_router import get_llm_router_chain
from langchain_ollama import ChatOllama
from langchain.agents import AgentType, initialize_agent


load_dotenv()

csv_agent = csv_tool_agent()
rag_chain = text_tool_agent()

#-------- Running miultiple tool agent with master agent using is too slow, to handle this slowness, I followed the different approach using llm router,. refer to run_router
def setup_master_agent():

    # Chat using ollama llama3.2 
    ollama_chat = ChatOllama(model="llama3.2", temperature=0, num_gpu=8)

    # Combine all the tools into a single list
    tools = [rag_chain, csv_agent]

    # Initialize the master agent with the list of tools and the Ollama model
    master_agent = initialize_agent(
        tools,
        ollama_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=100,
        early_stopping_method="generate"
    )
    return master_agent


# master_agent = setup_master_agent()


# Chat using ollama llama3.2 
ollama_chat = ChatOllama(model="llama3.2", temperature=0, num_gpu=8)
def run_router(query: str):
    router_chain = get_llm_router_chain(ollama_chat)
    route = router_chain.run(query).strip().lower()
    print(f"Routed to: {route}")
    
    if route == "csv":
        return csv_agent.invoke(query)
    elif route == "text":
        return rag_chain.invoke(query)
    else:
        return "Sorry, I couldn't determine how to handle your request."
    

def conversation_chat(message, history):
    # Note: Ignore the Gradio history, history is already handlled in RAG usign ConversationBufferMemory.
    # result = master_agent.run({"input":message})
    result = run_router(message)
    print('Received the response...')
    print(result)
    # Check for the correct key and return the string content
    if isinstance(result, dict) and 'output' in result:
        return result['output']
    elif isinstance(result, dict) and 'answer' in result:
        return result['answer']
    else:
        # Fallback in case the output is already a string (rare for agents)
        return str(result)



def render_mobile_shop_ui():
    gr.ChatInterface(
        fn=conversation_chat, 
        type="messages", 
        title="Welcome to Naveen's Mobile Shop").launch(share=True)


# response = conversation_chat('provide me realme mobile model details including memory and price and corresponding model name ?', '')
# print(response)
render_mobile_shop_ui()


# Use this command to run : python src/rag/multi_tool_agent/mobile_shop_master_agent.py 