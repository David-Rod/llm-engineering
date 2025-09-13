# LLM Engineering Course Exercises

Solutions and implementations from the [LLM Engineering: Master AI and Large Language Models](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/) Udemy course.

## Course Overview

This repository contains my exercise solutions and projects from the LLM Engineering course, focusing on:
- Building and deploying LLM applications
- Working with different models (OpenAI, Anthropic, Local models)
- Creating UI interfaces with Gradio
- Implementing streaming responses
- Managing system prompts and context

## Project Structure

```
llm-exercises/
├── week1/
├── week2/
├── week3/
├── week4/
├── week5/
├── week6/
└── README.md
```

## Exercise Solutions

### Week 1
- Basic LLM interactions
- System prompts
- Model selection

### Week 2
- Streaming implementations
- UI development with Gradio
- Multi-model support

### Week 3
- Test data generation
- Model fine-tuning
- Advanced prompting

### Week 4
- Compare open source vs closed source models
- Evaluate code generation
- Generate code for business tasks

### Week 5
- RAG fundamentals
- LangChain
- Vector Databases

### Week 6
- Fine Tuning Frontier Models
- Weights and biases jobs
- Processing and storage of training data as files
- **_NOTE_**: There is an issue with the approach for this week since the `*_lite.pkl` data files contain a poor distribution of pricing, skewing results to appear better than they are since item prices are mostly cheap. The fine tuned version of chat gpt also correctly predicted exact prices for test data indicating overtraining or training on exact subset of data.

## Usage

Each week's exercises are contained in their own directory with dedicated notebooks and Python files.

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Gradio
- OpenAI
- Anthropic


## Acknowledgments

- Course instructor and content creators
- OpenAI, Anthropic, and HuggingFace for model access
