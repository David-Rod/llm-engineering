import os
import io
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
# from google import genai
import google.generativeai as genai
from google.generativeai import types
import anthropic
from IPython.display import Markdown, display, update_display
import gradio as gr
import subprocess

from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'your-key-if-not-using-env')

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
os.environ['HF_QWEN_URL'] = os.getenv('HF_QWEN_URL', 'your-url-if-not-using-env')
os.environ['HF_CODE_GEMMA_URL'] = os.getenv('HF_CODE_GEMMA_URL', 'your-url-if-not-using-env')



CODE_QWEN_URL = os.environ['HF_QWEN_URL']
CODE_GEMMA_URL = os.environ['HF_CODE_GEMMA_URL']
HF_TOKEN = os.environ['HF_TOKEN']

openai = OpenAI()
claude = anthropic.Anthropic()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])


OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
GEMINI_MODEL = "gemini-2.5-flash-lite"

genai_model = genai.GenerativeModel(GEMINI_MODEL)
code_qwen = "Qwen/CodeQwen1.5-7B-Chat"
code_gemma = "google/codegemma-7b-it"

system_message = "You are a code translator which translates Python to C++ code. You only output raw code with no formatting, explanations, or markdown. Never use ``` code blocks. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time. Keep implementations of random number generators identical so that results match exactly."

def user_prompt_for(python):
    user_prompt = "CRITICAL REQUIREMENT: Respond ONLY with raw C++ code only. No explanations, no markdown formatting, no code blocks, no Python code remnants.\n\n"
    user_prompt += "Task: Rewrite this Python code in C++ with the fastest possible implementation that produces identical output.\n\n"
    user_prompt += "Please include all necessary dependencies such as <iomanip>\n\n"
    user_prompt += "Requirements:\n"
    user_prompt += "- Include all necessary #include statements\n"
    user_prompt += "- Make sure to include <chrono> if using timing, <iomanip> for formatting, and use std:: prefixes.\n"
    user_prompt += "- Add brief inline comments and docstring for clarity\n"
    user_prompt += "- NO markdown code blocks (```cpp)\n"
    user_prompt += "- NO explanatory text before or after code\n"
    user_prompt += "- NO token artifacts like </start_of_turn> or <|im_end|>\n\n"
    user_prompt += "Python code to convert:\n"
    user_prompt += python
    user_prompt += "\n\nOutput format: Raw C++ code starting with #include statements."
    user_prompt += "\n\nIMPORTANT: Your response should start with #include and contain ONLY C++ code. "
    user_prompt += "No markdown blocks, no explanations, no ```cpp formatting."
    return user_prompt

def gemma_user_prompt(message):
    return [
        {"role": "user", "content": user_prompt_for(message)}
    ]

sample_items = [
    {'name': 'UPLIFT V2 Standing Desk, 48" x 30" Bamboo Desktop', 'quantity': 2, 'price': '599.00'},
    {'name': 'Herman Miller Aeron Ergonomic Office Chair, Size B, Graphite', 'quantity': 4, 'price': '1395.00'},
    {'name': 'Dell UltraSharp 27" 4K USB-C Monitor (U2723QE)', 'quantity': 6, 'price': '649.99'},
    {'name': 'Amazon Basics Office Supply Bundle - Pens, Notebooks, Folders, Paper', 'quantity': 3, 'price': '49.99'},
    {'name': 'Logitech MX Keys Advanced Wireless Illuminated Keyboard', 'quantity': 6, 'price': '99.99'},
    {'name': 'Logitech MX Master 3S Advanced Wireless Mouse', 'quantity': 6, 'price': '99.99'},
    {'name': 'SteelSeries QcK Gaming Mouse Pad - Cloth Surface (Medium)', 'quantity': 6, 'price': '14.99'},
    {'name': 'VIVO Dual Monitor Desk Mount Stand for 13" to 27" Screens', 'quantity': 3, 'price': '39.99'},
    {'name': 'HON Brigade 4-Drawer Letter-Size File Cabinet, Light Gray', 'quantity': 2, 'price': '319.99'}
]

categorize_func = '''
import time
sample_items = [
    {'name': 'UPLIFT V2 Standing Desk, 48" x 30" Bamboo Desktop', 'quantity': 2, 'price': '599.00'},
    {'name': 'Herman Miller Aeron Ergonomic Office Chair, Size B, Graphite', 'quantity': 4, 'price': '1395.00'},
    {'name': 'Dell UltraSharp 27" 4K USB-C Monitor (U2723QE)', 'quantity': 6, 'price': '649.99'},
    {'name': 'Amazon Basics Office Supply Bundle - Pens, Notebooks, Folders, Paper', 'quantity': 3, 'price': '49.99'},
    {'name': 'Logitech MX Keys Advanced Wireless Illuminated Keyboard', 'quantity': 6, 'price': '99.99'},
    {'name': 'Logitech MX Master 3S Advanced Wireless Mouse', 'quantity': 6, 'price': '99.99'},
    {'name': 'SteelSeries QcK Gaming Mouse Pad - Cloth Surface (Medium)', 'quantity': 6, 'price': '14.99'},
    {'name': 'VIVO Dual Monitor Desk Mount Stand for 13" to 27" Screens', 'quantity': 3, 'price': '39.99'},
    {'name': 'HON Brigade 4-Drawer Letter-Size File Cabinet, Light Gray', 'quantity': 2, 'price': '319.99'}
]

def categorize_order_items(items):
    category_keywords = {
        'Furniture': [
            'desk', 'chair', 'table', 'cabinet', 'filing cabinet', 'stand',
            'furniture', 'seating', 'workstation', 'storage', 'shelf',
            'drawer', 'office furniture', 'ergonomic chair', 'standing desk'
        ],
        'Electronics': [
            'monitor', 'keyboard', 'mouse', 'computer', 'electronic', 'wireless',
            'usb', 'digital', 'screen', 'display', 'tech', 'device', 'gaming',
            'bluetooth', 'connectivity', 'hardware', 'peripheral'
        ],
        'Stationary': [
            'pen', 'pencil', 'paper', 'notebook', 'folder', 'stationary',
            'stationery', 'office supply', 'writing', 'notepad', 'binder',
            'clip', 'stapler', 'tape', 'supplies', 'bundle'
        ]
    }


    categorized_items = {
        'Furniture': [],
        'Electronics': [],
        'Stationary': []
    }


    for item in items:

        item_name = ''
        if isinstance(item, dict):
            item_name = (item.get('name', '') or
                        item.get('product_name', '') or
                        item.get('title', '') or
                        str(item)).lower()
        else:
            item_name = str(item).lower()


        category_scores = {}

        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in item_name:
                    score += len(keyword.split())
            category_scores[category] = score


        if max(category_scores.values()) > 0:
            best_category = max(category_scores, key=category_scores.get)
            categorized_items[best_category].append(item)
        else:

            categorized_items['Stationary'].append(item)

    return categorized_items

start_time = time.time()
result = categorize_order_items(sample_items)
end_time = time.time()

print(f"Execution Time: {(end_time - start_time):.6f} seconds")
for category, items in result.items():
    print("")  # Empty line before each category
    print("{} ({} items):".format(category, len(items)))
    for item in items:
        name = item["name"]
        print("  - {}".format(name))
'''

def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def write_output(cpp):
    code = cpp.replace("```cpp","").replace("```","")
    with open("optimized.cpp", "w") as f:
        f.write(code)

def stream_gpt(python):
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```cpp\n','').replace('```','')

def stream_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```cpp\n','').replace('```','')

def stream_gemini(python):
    response = genai_model.generate_content(
        user_prompt_for(python),
        stream=True
    )
    reply = ""
    for chunk in response:
        if chunk.text:
            fragment = chunk.text
            reply += fragment
            yield reply.replace('```cpp\n','').replace('```','')


def execute_python(code):
    try:
        namespace = {
            'time': __import__('time'),
            'json': __import__('json'),
            'sys': __import__('sys'),
            'dict': dict,
            'list': list,
            'str': str,
            'max': max,
            'len': len,
            'isinstance': isinstance,
            'print': print,
            'enumerate': enumerate
        }
        output = io.StringIO()
        sys.stdout = output
        compiled = compile(code, '<string>', 'exec')
        exec(compiled, namespace)
        return output.getvalue()
    finally:
        sys.stdout = sys.__stdout__


def execute_cpp(code):
    write_output(code)
    compile_cmd = ["g++", "-O3", "-std=c++17", "-march=x86-64-v3", "-mtune=native", "-o", "optimized", "optimized.cpp"]
    try:
        compile_result = subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
        run_cmd = ["./optimized"]
        run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
        return run_result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred:\n{e.stderr}"

def stream_code_qwen(python):
    tokenizer = AutoTokenizer.from_pretrained(code_qwen)
    messages = messages_for(python)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    client = InferenceClient(CODE_QWEN_URL, token=HF_TOKEN)
    stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
    result = ""
    for r in stream:
        result += r.token.text
        yield result

def stream_gemma(python):
    gemma_tokenizer = AutoTokenizer.from_pretrained(code_gemma)
    messages = gemma_user_prompt(python)
    text = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    client = InferenceClient(CODE_QWEN_URL, token=HF_TOKEN)
    stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
    result = ""
    for r in stream:
        result += r.token.text
        yield result

def clean_response(response):
    response = response.replace("```cpp", "")
    response = response.replace("```", "")
    response = response.replace("<|im_end|>", "")
    response = response.replace("</start_of_turn>", "")
    response = response.replace("<|im_start|>", "")

    return response.strip()

def optimize(python, model):
    if model=="GPT":
        result = stream_gpt(python)
    elif model=="Claude":
        result = stream_claude(python)
    elif model=="Gemini":
        result = stream_gemini(python)
    elif model=="Qwen2":
        result = stream_code_qwen(python)
    elif model=="Gemma":
        result = stream_gemma(python)
    else:
        raise ValueError("Unknown model")
    result = (clean_response(stream_so_far) for stream_so_far in result)
    for stream_so_far in result:
        yield stream_so_far

css = """
.python {background-color: #306998;}
.cpp {background-color: #050;}
"""

def select_sample_program(sample_program):
    if sample_program=="categorize_items":
        return categorize_func
    else:
        return "Type your Python program here"

import platform

VISUAL_STUDIO_2022_TOOLS = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\Tools\\VsDevCmd.bat"
VISUAL_STUDIO_2019_TOOLS = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\Common7\\Tools\\VsDevCmd.bat"

simple_cpp = """
#include <iostream>

int main() {
    std::cout << "Hello";
    return 0;
}
"""

def run_cmd(command_to_run):
    try:
        run_result = subprocess.run(command_to_run, check=True, text=True, capture_output=True)
        return run_result.stdout if run_result.stdout else "SUCCESS"
    except:
        return ""

def c_compiler_cmd(filename_base):
    my_platform = platform.system()
    my_compiler = []

    try:
        with open("simple.cpp", "w") as f:
            f.write(simple_cpp)

        if my_platform == "Windows":
            if os.path.isfile(VISUAL_STUDIO_2022_TOOLS):
                if os.path.isfile("./simple.exe"):
                    os.remove("./simple.exe")
                compile_cmd = ["cmd", "/c", VISUAL_STUDIO_2022_TOOLS, "&", "cl", "simple.cpp"]
                if run_cmd(compile_cmd):
                    if run_cmd(["./simple.exe"]) == "Hello":
                        my_compiler = ["Windows", "Visual Studio 2022", ["cmd", "/c", VISUAL_STUDIO_2022_TOOLS, "&", "cl", f"{filename_base}.cpp"]]

            if not my_compiler:
                if os.path.isfile(VISUAL_STUDIO_2019_TOOLS):
                    if os.path.isfile("./simple.exe"):
                        os.remove("./simple.exe")
                    compile_cmd = ["cmd", "/c", VISUAL_STUDIO_2019_TOOLS, "&", "cl", "simple.cpp"]
                    if run_cmd(compile_cmd):
                        if run_cmd(["./simple.exe"]) == "Hello":
                            my_compiler = ["Windows", "Visual Studio 2019", ["cmd", "/c", VISUAL_STUDIO_2019_TOOLS, "&", "cl", f"{filename_base}.cpp"]]

            if not my_compiler:
                my_compiler=[my_platform, "Unavailable", []]

        elif my_platform == "Linux":
            # Try g++ first with x86-64-v3 architecture
            if os.path.isfile("./simple"):
                os.remove("./simple")
            compile_cmd = ["g++", "-O3", "-std=c++17", "-march=x86-64-v3", "-mtune=native", "-o", "simple", "simple.cpp"]
            if run_cmd(compile_cmd):
                if run_cmd(["./simple"]) == "Hello":
                    my_compiler = ["Linux", "GCC (g++)", ["g++", "-O3", "-std=c++17", "-march=x86-64-v3", "-mtune=native", "-o", f"{filename_base}", f"{filename_base}.cpp"]]

            # Try clang++ if g++ fails
            if not my_compiler:
                if os.path.isfile("./simple"):
                    os.remove("./simple")
                compile_cmd = ["clang++", "-O3", "-std=c++17", "-march=x86-64-v3", "-mtune=native", "-o", "simple", "simple.cpp"]
                if run_cmd(compile_cmd):
                    if run_cmd(["./simple"]) == "Hello":
                        my_compiler = ["Linux", "Clang++", ["clang++", "-O3", "-std=c++17", "-march=x86-64-v3", "-mtune=native", "-o", f"{filename_base}", f"{filename_base}.cpp"]]

            if not my_compiler:
                my_compiler=[my_platform, "Unavailable", []]

        elif my_platform == "Darwin":
            if os.path.isfile("./simple"):
                os.remove("./simple")
            compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", "simple", "simple.cpp"]
            if run_cmd(compile_cmd):
                if run_cmd(["./simple"]) == "Hello":
                    my_compiler = ["Macintosh", "Clang++", ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", f"{filename_base}", f"{filename_base}.cpp"]]

            if not my_compiler:
                my_compiler=[my_platform, "Unavailable", []]
    except:
        my_compiler=[my_platform, "Unavailable", []]

    if my_compiler:
        return my_compiler
    else:
        return ["Unknown", "Unavailable", []]


compiler_cmd = c_compiler_cmd("optimized")

with gr.Blocks(css=css) as ui:
    gr.Markdown("## Convert code from Python to C++")
    with gr.Row():
        python = gr.Textbox(label="Python code:", value=categorize_func, lines=10)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        with gr.Column():
            sample_program = gr.Radio(["categorize_items"], label="Sample program", value="categorize_items")
            model = gr.Dropdown(["GPT", "Claude", "Gemini", "Qwen2", "Gemma"], label="Select model", value="Claude")
        with gr.Column():
            architecture = gr.Radio([compiler_cmd[0]], label="Architecture", interactive=False, value=compiler_cmd[0])
            compiler = gr.Radio([compiler_cmd[1]], label="Compiler", interactive=False, value=compiler_cmd[1])
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Python")
        if not compiler_cmd[1] == "Unavailable":
            cpp_run = gr.Button("Run C++")
        else:
            cpp_run = gr.Button("No compiler to run C++", interactive=False)
    with gr.Row():
        python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
        cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

    sample_program.change(select_sample_program, inputs=[sample_program], outputs=[python])
    convert.click(optimize, inputs=[python, model], outputs=[cpp])
    python_run.click(execute_python, inputs=[python], outputs=[python_out])
    cpp_run.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(debug=True)

