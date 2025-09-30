from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

router_prompt = PromptTemplate.from_template("""
You are a router. Decide whether the user query is about a CSV table, or a text document.

If the query is about mobile model list, mobile model with "Memory, Storage, Rating, Selling Price and Original Price", say "csv".
If the query is about a text document, say "text".
Only respond with one word: "csv" or "text".

Query: {query}
Answer:""")

def get_llm_router_chain(llm):
    router_chain = LLMChain(llm=llm, prompt=router_prompt)
    print('get_llm_router_chain')
    return router_chain

