
"""
@author: Naveen N G
@date: 29-09-2025
@description: Text tool agent.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import glob
from typing import List
import pandas as pd
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.agents import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
load_dotenv()

database_name = 'multi_agent_vector_db'

def text_tool_agent():
    data_folder_path = os.path.join(parent_dir, "mobile_shop", "data", "company")
    folders = glob.glob(f"{data_folder_path}/*")
    all_documents = []

    for file in folders:
        text_loader = TextLoader(file)
        document = text_loader.load()
        all_documents.extend(document)

    # Split the document into chunks    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    processed_documents_chunk = text_splitter.split_documents(all_documents)
    
    # Chat using ollama llama3.2 
    ollama_chat = ChatOllama(model="llama3.2", temperature=0)

    # Memory to store the chat history for the context
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Initialize the Ollama embeddings model for RAG
    ollama_embeddings = OllamaEmbeddings(model="llama3.2", num_gpu=8)

    # Create a vector store and retriever using the Ollama embeddings
    database_path = os.path.join(current_dir, database_name)
    if os.path.exists(database_path):
        # If the database already exists, cleaning the database so that it won't add the same thing to the database
        Chroma(persist_directory=database_path, embedding_function=ollama_embeddings).delete_collection()
    vector_store = Chroma.from_documents(documents = processed_documents_chunk, embedding=ollama_embeddings, persist_directory=database_path)
    text_retriever = vector_store.as_retriever(search_kwargs={"k": 50})

    # putting it all together, set up the RAG conversation chain
    conversation_rag_chain = ConversationalRetrievalChain.from_llm(llm=ollama_chat, retriever=text_retriever, memory=memory)

    # Create a RetrievalQA chain for the text data, using the Ollama chat model
    # conversation_rag_chain = RetrievalQA.from_chain_type(
    #     llm=ollama_chat,
    #     chain_type="stuff",
    #     retriever=text_retriever,
    # )

    # Wrap the ConversationalRetrievalChain chain as a single tool
    text_tool = Tool(
        name="company_info_tool",
        func=conversation_rag_chain.invoke,
        description="""
            This tool is useful for general questions about the company, its history, mission, and leadership, company summary.
            The input should be a full question about general company information.
            Repond only based on the context provided, do not suggest anything before the context.
            Do not make anything up if you haven't been provided with relevant context
            """,
    )

    print('Text tool agent is ready..')
    return text_tool


# text_tool_agent()
