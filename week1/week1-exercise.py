# %% [markdown]
# # End of week 1 exercise
#
# To demonstrate your familiarity with OpenAI API, and also Ollama, build a tool that takes a technical question,
# and responds with an explanation. This is a tool that you will be able to use yourself during the course!

# %%
# imports
import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# %%
# set up environment
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")

# constants
MODEL_GPT = 'gpt-4o-mini'
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_LLAMA = 'llama3.2:1b'
HEADERS = {"Content-Type": "application/json"}
openai = OpenAI()

# %%
# here is the question; type over this to ask something new

question = "I would like to know what are the most important aspects of learning Italian, and would like for you to construct \
    a curriculum to help me speak the language fluently in the next 6 months. Please design curriculum according the the latest \
    research on language mastery, allowing me to have a whole understanding of the language unlike the approach taken by modern \
    language learning applicaitons."

user_prompt = "Please give a detailed explanation to the following question: " + question

system_prompt = "You are an assistant that the questions presented by a user and generates thoughtful analysis \
and creates concise and clear summaries that are organized by category of relevant information. Respond in markdown.\
Please include descriptive examples and be sure to frame the response in the context of the quetion provided."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# %%
# Get gpt-4o-mini to answer, with streaming
stream = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=messages,
        stream=True,
    )
response = ""
display_handle = display(Markdown(""), display_id=True)

dir(stream)
for chunk in stream:
    response += chunk.choices[0].delta.content or ''
    response = response.replace("```","").replace("markdown", "")
    update_display(Markdown(response), display_id=display_handle.display_id)

display(Markdown(response))

# %%
# Get Llama 3.2 to answer
payload = {
        "model": MODEL_LLAMA,
        "messages": messages,
        "stream": True
    }
ollama_response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)

# %%
# Convert byte string repsonse to single markdown string response
string_response = ollama_response.content.decode('utf-8')
json_objects = []

for line in string_response.strip().split('\n'):
    if line.strip():
        json_object = json.loads(line)
        json_objects.append(json_object)

combined_message_str = ""
for json_obj in json_objects:
    combined_message_str+=json_obj['message']['content']

display(Markdown(combined_message_str))


# %%
import ollama

ollama_response_2 = ollama.chat(model=MODEL_LLAMA, messages=messages)

# %%
display(Markdown(ollama_response_2.message.content))


