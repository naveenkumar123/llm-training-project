"""
@author: Naveen N G
@date: 28-09-2025
@description: Mobile shop application using RAG technique pipline using langchain
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gradio as gr
from mobile_shop.langchain_vector_db import get_shop_vector_data_store
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler


def create_rag_pipline():
    # Chat using ollama llama3.2 
    llm = ChatOllama(temperature=0.7, model="llama3.2")

    # Memory to store the chat history for the context
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

   # the retriever is an abstraction over the VectorStore that will be used during RAG
    vectorstore = get_shop_vector_data_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 149})

    # putting it all together, set up the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

    return conversation_chain

def conversation_chat(message, history):
    # Note: Ignore the Gradio history, history is already handlled in RAG usign ConversationBufferMemory.
    result = conversation_chain.invoke({"question":message})
    return result["answer"]


conversation_chain = create_rag_pipline()
print('Pipeline chain received..')

def render_chat_ui():
    gr.ChatInterface(
        fn=conversation_chat, 
        type="messages", 
        title="Welcome to Naveen's mobile shop").launch(share=True)
    
render_chat_ui()

# Use this command to run : python src/rag/mobile_shop_rag_pipeline.py 