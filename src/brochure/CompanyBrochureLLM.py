from openai import OpenAI
from IPython.display import Markdown, display, update_display
import json
import os
from dotenv import load_dotenv
from brochure.WebsiteTransform import Website


def get_system_prompt() -> str:
    link_system_prompt = """
    You are provided with a list of links found on a webpage.
    You are able to decide which of the links would be most relevant to include in a brochure about the company, 
    such as links to an About page, or a Company page, or Careers/Jobs pages.
    You should respond in JSON as in this example:
    """
    link_system_prompt+= """
    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page", "url": "https://another.full.url/careers"}
        ]
    }
    """
    return link_system_prompt



def get_links_user_prompt(website) -> str:
    user_prompt=f"Here is the list of links on the website of {website.url} -"
    user_prompt+=  "please decide which of these are relevant web links for a brochure about the company, respond with full https URL in JSON format. \
                    Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt



def get_links(website: Website) -> List[str]:
    load_dotenv()
    openai = OpenAI()
    respone = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_links_user_prompt(ed)}  
        ],
        response_format={"type": "json_object"},
    )
    result = respone.choices[0].message.content
    return json.loads(result)


# ed = Website("https://edwarddonner.com")
# print(ed.links)
# print(get_links(ed))

# Get all the details of the company form the main website and which also brings the content of the links in the website present. 
def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(Website(url)) # This makes a call to the OpenAI API LLM call.
    for link in links:
        result += f"Link type: {link['type']}\n"
        result += Website(link['url']).get_contents()
    return result

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt


def create_company_brochure(company, url):
    system_prompt = """
        You are an assistant that analyzes the contents of several relevant pages from a company website 
        and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.
        Include details of company culture, customers, company products and careers/jobs if you have the information.
    """
    load_dotenv()
    openapi = OpenAI()
    response = openapi.chat.completions.create(
        mpdel="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company, url)}
        ]
    )
    result = response.choices[0].message.content
    display(Markdown(result))

def ceate_stream_company_brochure(company, url):
    system_prompt = """
        You are an assistant that analyzes the contents of several relevant pages from a company website 
        and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.
        Include details of company culture, customers, company products and careers/jobs if you have the information.
    """
    load_dotenv()
    openapi = OpenAI()
    streamResponse = openapi.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company, url)}
        ],
        stream=True
    )
    display_handle = display(Markdown(""), display_id=True)
    response = ""
    for chunk in streamResponse:
        response += chunk.choices[0].delta.content or ''  # save the event response
        response = response.replace("```","").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)



# create_company_brochure("Airbus", "https://www.airbus.com")
# ceate_stream_company_brochure("Airbus", "https://www.airbus.com")