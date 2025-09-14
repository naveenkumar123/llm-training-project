
"""
@author: Naveen N G
@date: 14-09-2025
@description: This module provides functionality to Book the ticket for Nee Airlines using LLMs (Large Language Models) 
            such as OpenAI's GPT and Llama. It uses tool calling feature of LLMs to get the ticket price for a destination.
            The module includes a Gradio UI for interactive booking assistant.
"""



import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import os
from dotenv import load_dotenv
import gradio as gr
from llmchat.LlamaChat import LlamaChat
from llmchat.GptChat import GptChat

def get_system_prompt() -> str:
   
    system_prompt = """
    You are a helful assistant for an Airline bookin service of Nee Airlines.
    You are provided with the following information about Nee Airlines:
    1. Nee Airlines operates flights to various domestic and international destinations.
    2. The airline offers different classes of service, including Economy, Business, and First Class.
    3. Passengers can book one-way or round-trip tickets.
    4. The airline provides options for additional services such as extra baggage, in-flight meals, and seat selection.
    5. Customers can manage their bookings online, including changing flight dates and canceling tickets.
    6. Nee Airlines has a frequent flyer program that rewards loyal customers with points that can be redeemed for flights and upgrades.
    7. The airline has a customer support team available 24/7 to assist with any inquiries or issues.
    8. Safety and comfort are top priorities for Nee Airlines, with modern aircraft and attentive service.
    9. The airline offers special discounts and promotions during certain times of the year.
    10. Customers can book flights through the airline's website, mobile app, or authorized travel agents.
    11. Nee Airlines has partnerships with other airlines, allowing for code-sharing and seamless travel experiences.
    12. The airline adheres to all international aviation regulations and standards.
    13. Passengers are required to check in at least 2 hours before domestic flights and 3 hours before international flights.  
    14. Provide link for booking : https://www.google.in/
    Give short, courteous answers, no more than 1 sentence and assume you are providing the real time information.
    Always be accurate. If you don't know the answer, say so."""
   
    # system_prompt = "You are a helpful assistant for an Airline called FlightAI. "
    # system_prompt += "Give short, courteous answers, no more than 1 sentence. "
    # system_prompt += "Always be accurate. If you don't know the answer, say so."

    return system_prompt


ticket_price_info = {"bangalore": "2000 INR", "chennai": "1000 INR", "mumbai": "2500 INR", "delhi": "5000 INR",
                     "new york": "80000 INR", "london": "50000 INR", "paris": "40000 INR", "tokyo": "10000 INR",
                     "berlin": "45000 INR", "sydney": "60000 INR", "singapore": "10000 INR", "dubai": "15000 INR"}

def get_ticket_price(destination : str) -> str:
    print(f"Tool get_ticket_price called for {destination}")
    city = destination.lower()
    return ticket_price_info.get(city, "Unknown")


# This price function can be called by the LLM using the tool calling feature.
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "The destination city to get the ticket price for."
            }
        },
        "required": ["destination"],
        "additionalProperties": False
    }
}
# used for tool calling in LLM
tools = [{'type': 'function', 'function': price_function}]

# Multi model chat conversation with tool calling, uncomment for Ollma LlamaChat
def chat_conversation(message, chat_history):
    load_dotenv()
    local_system_prompt = get_system_prompt()

    messages = [{"role": "system", "content": local_system_prompt}] + chat_history + [{"role": "user", "content": message}]
    
    response = GptChat().chat_with_tool(messages, tools=tools)
    # response = LlamaChat().chat_with_tool(messages, tools=tools, stream=False)
    # If the response indicates a tool call, handle it

    if response.choices[0].finish_reason=="tool_calls":
    # if  response['message']['tool_calls']:
        print("Model requested tool calls.")
        # message = response.message
        message = response.choices[0].message
        response, destination = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        # response = LlamaChat().chat_result(messages)
        response = GptChat().chat_result(messages)
    else:
        print("Model did not request tool calls.")  
    # return response.message.content 
    return response.choices[0].message.content 

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    # Below line for Ollama LlamaChat
    # destination = tool_call.function.arguments.get("destination", "")
    arguments = json.loads(tool_call.function.arguments)
    destination = arguments.get('destination')
    price = get_ticket_price(destination)
    response = {
        "role": "tool",
        "content": json.dumps({"destination": destination,"price": price}),
         "tool_call_id": tool_call.id # Comment this line for Ollama LlamaChat
    }
    return response, destination


def render_booking_chatbot_ui():
    gr.ChatInterface(
        fn=chat_conversation, 
        type="messages", 
        title='Nee Airlines Booking Assistant').launch(share=True)
    

render_booking_chatbot_ui()
# To run this file, use the command: python src/airline_chatbot/NeeAirlineBooking.py


