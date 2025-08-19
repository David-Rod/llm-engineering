# %%
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from pathlib import Path

from bs4 import BeautifulSoup
import imaplib
import yaml
import logging
import pandas as pd
import json
import email

from email.parser import Parser

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings

parser = Parser()

# %%
db_name = "email_vector_db"
MODEL = "gpt-4o-mini"

load_dotenv('../.env/.env-config', override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
print("OpenAI API Key:", os.environ['OPENAI_API_KEY'][:10] + '...')


# %% [markdown]
# # Setup and Imports
# Run this cell first to make all imports available for the rest of the notebook.

# %%
def load_credentials(filename, config_dir='.env'):
    current_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    filepath = current_dir.parent / config_dir / filename

    try:
        if not filepath.exists():
            raise FileNotFoundError(f"Credentials file not found at: {filepath}")

        with open(filepath, 'r') as file:
            credentials = yaml.safe_load(file)

            if not isinstance(credentials, dict):
                raise ValueError("Credentials file must contain a YAML dictionary")

            if 'user' not in credentials or 'password' not in credentials:
                raise KeyError("Credentials file must contain 'user' and 'password' fields")

            return credentials['user'], credentials['password']

    except Exception as e:
        logging.error(f"Failed to load credentials from {filepath}: {e}")
        raise

# %%
def parse_email_from_string(email_string):
    email_message = parser.parsestr(email_string)
    return email_message

# %%
def connect_to_gmail_imap(*credentials) -> imaplib.IMAP4_SSL:
    email, app_token = credentials
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(email, app_token)
    return mail

# %%
def write_message_to_file(message_numbers, filename='knowledge-base/email-file.txt'):

    file_path = Path(filename)
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted {file_path}")
    else:
        print(f"File {file_path} does not exist")

    for num in message_numbers[0].split():
        typ, msg_data = mail.fetch(num, '(RFC822)')
        emails = parse_email_from_string(msg_data[0][1].decode("utf-8"))
        payload = emails.get_payload()[0].get_payload()

        if not isinstance(payload, list):
            content = payload
        else:
            content = ''

        cleantext = BeautifulSoup(content, "html.parser").text
        with open(filename, 'a') as f:
            f.write(f"\n--- Email {num.decode()} ---\n")
            f.write(str(cleantext))
            f.write("\n" + "="*50 + "\n")

# %%
credentials = load_credentials('credentials.yaml')
mail = connect_to_gmail_imap(*credentials)
mail.select('INBOX')
mail.select('"[Gmail]/All Mail"')  # This gets ALL emails in your account
status, message_numbers = mail.search(None, 'FROM "amazon.com"')
write_message_to_file(message_numbers)
# status, folders = mail.list()
# print("Available folders:")
# for folder in folders:
#     folder_name = folder.decode('utf-8')
#     print(f"  {folder_name}")



# %%
text_files = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}
documents = []

for file_path in text_files:
    loader = TextLoader(file_path, **text_loader_kwargs)
    doc = loader.load()
    documents.extend(doc)

text_splitter = CharacterTextSplitter(chunk_size=750, chunk_overlap=100, separator="\n\n", strip_whitespace=True )
chunks = text_splitter.split_documents(documents)
print(f"Total number of chunks: {len(chunks)}")

# %%
embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# %%
collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

# %%
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# Alternative - if you'd like to use Ollama locally, uncomment this line instead
# llm = ChatOpenAI(temperature=0.7, model_name='llama3.2', base_url='http://localhost:11434/v1', api_key='ollama')

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# %%
def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

# %%
# Please tell me if my Amazon emails contain information about a portable monitor
# Please tell me if my Amazon emails contain information about a Newporter Classic guitar
# Can you provide order details from that order based on my Amazon emails?

view = gr.ChatInterface(chat, type="messages").launch()


