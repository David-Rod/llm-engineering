
# imports
import os
import requests
import json
from typing import List
from dotenv import load_dotenv
import base64
from io import BytesIO

from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
import ipywidgets as widgets

import anthropic
from openai import OpenAI
import whisper

import gradio as gr



load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:6]}")

else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API key exists and begins with {anthropic_api_key[:6]}")
else:
    print("Anthropic API Key not set")


GPT_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
OLLAMA_MODEL = "llama3.2:1b"
AUDIO_MODEL = "tts-1"
openai = OpenAI()
claude = anthropic.Anthropic()


translation_message = "You are a helpful assistant. Your primary role is to translate the user's input from English to Italian, or from Italan to English. "
translation_message += "You will receive a message from the user, and you should respond to user input translated in the appropriate language. "
translation_message += "If the user prompt input is in Italian, response in English, or reply with 'I don't understand (non capisco). "
translation_message += "If the input is in English, respond in Italian, or reply with 'Non capisco (I don't understand)'. Examples: "
# Multishot prompting
translation_message += "If the user says 'Hello, how are you?', respond with 'Ciao, come stai?' or a with a standard reply to the question or statement. "
translation_message += "If the user says 'Ciao, come stai?', respond with 'Hello, how are you?' or a with a standard reply to the question or statement. "
translation_message += "If the user says 'Non capisco', respond with 'I don't understand'. "
translation_message += "If the user says 'I don't understand', respond with 'Non capisco'. "
translation_message += "If the user says please translate 'Hello, how are you?' to Italian, respond with 'Ciao, come stai?' or a with a standard reply to the question or statement. "
translation_message += "If the user says please translate 'Ciao, come stai?' to English, respond with 'Hello, how are you?' or a with a standard reply to the question or statement. "





def stream_claude(prompt):
    result = claude.messages.stream(
    model=CLAUDE_MODEL,
    max_tokens=200,
    temperature=0.7,
    system=translation_message,
    messages=[
        {"role": "user", "content": prompt},
    ],
    )

    with result as stream:
        for text in stream.text_stream:
               yield text

def stream_gpt(prompt):
    stream = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": translation_message},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    for chunk in stream:
        text = chunk.choices[0].delta.content or ''
        if text:
            yield text

def stream_llama(prompt):
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    stream = ollama_via_openai.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": translation_message},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content or ''
        if text:
            yield text


def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    elif model == "Llama":
        result = stream_llama(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


from pydub import AudioSegment
from pydub.playback import play

openai.audio.speech

def talker(message):
    response = openai.audio.speech.create(
      model=AUDIO_MODEL,
      voice="shimmer",
      input=message
    )

    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


def translate_text():
    model = whisper.load_model("small")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("audio.mp3")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)


tools = [{"type": "function", "function": translate_text}]

def handle_tool_call(prompt):
    if "translate" in prompt.lower():
        response = translate_text(prompt)
    else:
        response = "I don't understand the tool call."

    return response

with gr.Blocks() as ui:
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
        model_selector = gr.Dropdown(["Claude", "GPT", "Llama"], label="Select model", value="Claude")
    with gr.Row():
        chatbot = gr.Chatbot(height=207, type="messages")
        translation_output = gr.Textbox(label="Translation Output", placeholder="Translated text will appear here", lines=7)

    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history, model):
        if model == "Claude":
            translation = "".join(stream_claude(message))
        elif model == "GPT":
            translation = "".join(stream_gpt(message))
        elif model == "Llama":
            translation = "".join(stream_llama(message))
        else:
            translation = "Unknown model"
        history += [{"role": "user", "content": message}]
        talker(translation)

        return "", history, translation


    entry.submit(
        do_entry,
        inputs=[entry, chatbot, model_selector],
        outputs=[entry, chatbot, translation_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch()


