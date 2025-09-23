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
├── week7/
├── week8/
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

### Week 7: Advanced Fine-Tuning and Model Training
- **Custom Model Training**: Train open-source models from scratch or continue pre-training
- **Parameter-Efficient Fine-Tuning**: Implement LoRA and QLoRA for memory-efficient training
- **Model Repository Management**: Upload trained models to HuggingFace Hub with proper documentation
- **Training Monitoring**: Track experiments with Weights & Biases (wandb) for metrics visualization
- **Hyperparameter Optimization**: Experiment with learning rates, batch sizes, and training strategies
- **Model Evaluation**: Compare custom-trained models against baseline models using standardized benchmarks

### Week 8: Multi-Agent Systems and Production Deployment
- **Capstone Project**: End-to-end price monitoring and notification system
- **Multi-Agent Architecture**: Design and implement 7 specialized agents working in coordination:
  - **UI Agent**: Gradio-based user interface for system interaction
  - **Orchestration Agent**: Memory management and system-wide logging
  - **Planning Agent**: Strategic coordination of agent activities and workflow management
  - **Scanner Agent**: Web scraping and deal identification across multiple platforms
  - **Ensemble Agent**: Price prediction using multiple fine-tuned models for accuracy
  - **Messaging Agent**: Real-time push notifications and user communication
  - **Data Agent**: Price history storage and trend analysis
- **Production Considerations**: Type hinting, Comments, Error handling, Logging
- **Real-World Integration**: API integrations, data pipelines, and system reliability
- **Modal for Serverless Platforming**: Deploy production models

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
