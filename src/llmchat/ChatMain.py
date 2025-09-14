"""
@author: Naveen N G
@date: 10-09-2025
@description: Main interface to chat with different LLM models (GPT, Llama, etc.)
"""

from GptChat import GptChat
from LlamaChat import LlamaChat

class ChatMain:
    def __init__(self, model_type: str = "gpt", model_name: str = "gpt-4o-mini",):
        if model_type == "gpt":
            self.chat_model = GptChat(model=model_name)
        elif model_type == "llama":
            self.chat_model = LlamaChat(model=model_name)
        else:
            raise ValueError("Unsupported model type. Choose 'gpt' or 'llama'.")

    def chat(self, messages):
        return self.chat_model.chat(messages)
    

# Example usage:
gpt_system_prompt = "You are a helpful assistant."
llama_system_prompt = "You are a helpful assistant."

gpt_messages = ['Hi']
llama_messages= ['Hi there']

def call_gpt():
    gptChat = ChatMain(model_type="gpt", model_name="gpt-4o-mini")
    gpt_messages_prompt = [{"role": "system", "content": gpt_system_prompt}]
    for gpt, llama in zip(gpt_messages, llama_messages):
        gpt_messages_prompt.append({"role": "assistant", "content": gpt})
        gpt_messages_prompt.append({"role": "user", "content": llama})
    return gptChat.chat(gpt_messages_prompt) 


def call_ollama():
    llamaChat = ChatMain(model_type="llama", model_name="llama3.2")
    llama_messages_prompt = [{"role": "system", "content": llama_system_prompt}]
    for gpt, llama in zip(gpt_messages, llama_messages):
        llama_messages_prompt.appemd({"role": "assistant", "content": llama})
        llama_messages_prompt.appemd({"role": "user", "content": gpt})
    return llamaChat.chat(llama_messages_prompt)


print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Ollama:\n{llama_messages[0]}\n")

# Continue the conversation till 5 exchanges
for i in range(5):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)
    
    ollama_next = call_ollama()
    print(f"Llama:\n{ollama_next}\n")
    llama_messages.append(ollama_next)