"""
@author: Naveen N G
@date: 21-09-2025
@description: This application provide the real time all the Upcoming IPO's in NSE/BSE and provides the IPO Details and Financial infromation about the company.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
import requests
from llmchat.GptChat import GptChat
from llmchat.LlamaChat import LlamaChat
from IPython.display import Markdown, display
import gradio as gr
import threading
import fitz  # PyMuPDF

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

load_dotenv()

# Note: connect to this indianapi.in to get the API token
ipo_url = "https://stock.indianapi.in/ipo"
IPO_API_KEY = os.getenv('IPO_API_KEY')
headers = {"X-Api-Key": IPO_API_KEY}

ipo_companies = {}
api_response = []


def pdf_to_text(pdf_content_bytes):
    try:
        # Load the PDF content from bytes
        pdf_document = fitz.open(stream=pdf_content_bytes, filetype="pdf")
        
        text_content = ""
        # Iterate through each page and extract text
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()
            
        return text_content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def reset_api_call():
    global api_response
    api_response.clear()
     # Schedule the function to run again in 1 hour (3600 seconds)
    threading.Timer(3600, reset_api_call).start()
# Initial call to start the process
reset_api_call()


# API call to get all IPO's list
def api_call():
    if len(ipo_companies) == 0:   
        response = requests.get(ipo_url, headers=headers)
        print(f"Response received from {ipo_url}")
        return response.json()
    else:
        print("External API call is not made, previous data exists.")

# Extract the company name and document
def company_list():
    response = api_call()
    companies = response.get('upcoming')
    for company_info in companies:
        company_name = company_info.get('name')
        document_url = company_info.get('document_url') or 'Link Not found'
        status = company_info.get('status')
        last_date_to_sumit = company_info.get('listing_date') or ''
        bidding_start_date = company_info.get('bidding_start_date') or ''
        min_price = company_info.get('min_price') or ''
        max_price = company_info.get('max_price') or ''
        if status == 'upcoming':
            company_details = {}
            company_details['document_url']  = document_url
            company_details['last_date_to_sumit']  = last_date_to_sumit
            company_details['bidding_start_date'] = bidding_start_date
            company_details['min_price'] = min_price
            company_details['max_price'] = max_price
            ipo_companies[company_name] = company_details

# Get the company names list
def company_names():
    return list(ipo_companies.keys())

max_retry = 3
# Get the company document details
def document_content(documentUrl):
    headers = {
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }

    content = ''
    retry = 0
    max_retry = 3
    while retry <= max_retry:
        try:
            response = requests.get(documentUrl, headers)
            content_stream =  response.content
            content = pdf_to_text(content_stream)
            # print(len(content))
            # if len(content) > 300000:
            #     return content[0:10000]
            # else:
            #     return content
            return content
        except Exception as e:
            print(f'failed to get the document content, retrying attempt : {retry}')
        retry += 1
    
    return 'No document found for the company'

# LLM analysis..
def ipo_company_summary(company_name):
    company_info = ipo_companies[company_name]
    print(company_info) 
    documentUrl = company_info.get('document_url')

    print(f"selected company:{company_name} and the coresponding document url:{documentUrl}")
    
    # Skiping for now, huge amount of data in the pdf, around 400 poages, which is too much for each call to summerize.
    # company_information = document_content(documentUrl)

    system_prompt = """
                    You are a financial analyst specialized in Indian equities (NSE/BSE). Your task is to analyze the company information provided by the user (financial statements, press releases, investor presentations, earnings transcripts, filings, news excerpts, etc.) and produce a concise, factual company summary focused on:
                    - IPO Details (example:  Bidding start date, Last date to submit, Max price and Min price)
                    - Financial performance
                    - Company growth in percentage
                    - Year-on-year (YoY) growth details
                    - Main products/segments
                    - Sales details (segment/geography/channel, as available)
                """


    user_prompt = f"You are looking at a company : {company_name}\n"
    user_prompt += f"""
                Please analyze the provided company information and return:
                - IPO Details
                - Company Fincaincial details
                - Financial Performance table
                - Growth Analysis (YoY, QoQ if quarterly, and 3Y/5Y CAGR if data allows)
                - Main Products and Segments
                - Sales Details (segment/geography/channel tables)
                - Profitability & Balance Sheet Highlights
                - Outlook/Guidance and Risks
                - Data Sources and Notes
                - Highlight the IPO submition document Link (example : Here is the IPO document link which is submitted to NSE/BSE: 'https://localhost:8000/document.pdf')
                - Do not mention the Note about system prompt.

                Assume INR crore, prefer consolidated figures, and follow the calculation and rounding rules provided in the system prompt \n
                Here is the IPO Details: \n
                Bidding start date : {company_info.get('bidding_start_date')}, Last date to submit: {company_info.get('last_date_to_sumit')}, 
                Max price : {company_info.get('max_price')}, Min price:  {company_info.get('min_price')} \n\n
                Below is the company document link provided to the NSE/BSE for company's complete Information.\n\n
                {documentUrl}
            """
    
    messages = [
        { "role" : "system", "content": system_prompt },
        { "role" : "user", "content": user_prompt }
    ]

    # streamResponse = GptChat('gpt-4.1-mini').chat(messages, True)
    # response = ""
    # for chunk in streamResponse:
    #     response += chunk.choices[0].delta.content or ''  # save the event response
    #     response = response.replace("```","").replace("markdown", "")
    #     # update_display(Markdown(response), display_id=display_handle.display_id)
    #     yield response

    streamResponse = LlamaChat().chat(messages, True)
    response = ""
    for chunk in streamResponse:
        response += chunk.message.content or ''
        response = response.replace("```","").replace("markdown", "")
        yield response

css = """
.scrollbox {
    max-height: 100px !important;;
    overflow-y: scroll !important;;
    border: 1px solid #e0e0e0 !important;;
    padding: 10px !important;;
    border-radius: 5px !important;;
}
"""

def render_ui(css=css):
    if IPO_API_KEY is None or not IPO_API_KEY:
        print('IPO API Key is missing..')

    with gr.Blocks() as ui:
        gr.Markdown("## Upcoming IPO Company details:")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                company_name = gr.Dropdown(company_names(), label="Select comapny")
                get_company_summary = gr.Button("Get Financial summary")
            with gr.Column(scale=2, min_width=300):
                company_summary = gr.Markdown(label="Company Financial Details", elem_classes=["scrollbox"])
        get_company_summary.click(ipo_company_summary, inputs=[company_name], outputs=[company_summary])
    ui.launch(inbrowser=True)


 # Get the companies list by default
company_list()

# Launch the application
render_ui()


# Use this command to run : python src/financial/ipo_companies_report.py 