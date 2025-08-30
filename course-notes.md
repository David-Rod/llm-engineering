# LLM Course Notes

## Model Categories

### Closed-Source Frontier Models
- **GPT**
- **Gemini**
- **Claude**

### Open-Source Frontier Models
- **Llama**
- **Qwen**
- **Gemma**

## Usage Methods
1. **Chat Interfaces**
   - Direct interaction like ChatGPT

2. **Cloud APIs**
   - LLM API using API key for code integration
   - Includes managed AI cloud services

3. **Direct Inference**
   - Local execution via HuggingFace Transformers with Ollama
   - Directly tokenizes input to model for weight assignment

## Frontier Model Platforms
- **OpenAI**: ChatGPT
- **Anthropic**: Claude
- **Google**: Gemini
- **Cohere**: Command R+
- **Meta**: LLama
- **Perplexity**: Perplexity

### Capabilities
#### Strengths
- Synthesizing information
- Iterative building
- Coding

#### Limitations
- Specialized domains (not PhD level but improving)
- Recent events (limited by knowledge cutoff date)
- Hallucinations (confidently make mistakes without acknowledging errors)

## Technical Components

### Transformers
- Changes input sequence into output sequence
- Tracks relationship between sequence components, like word association in prompts
- Processes long sequences in parallel
- Enables models to process long-range dependencies using self-attention
- Scales effectively with data and compute resources
- Powers most state-of-the-art LLMs (GPT, BERT, Claude, LLaMA)
- Foundational architecture enabling language generation, understanding, and reasoning

### GPT Tokenization Evolution
- Early development: character level training
- Intermediate: word-based training with vocabulary
- Current: Tokenized word chunks with elegant stem handling
- Reference: https://platform.openai.com/tokenizer
- Metrics:
  - 1 token ≈ 4 characters or ~.75 words (GPT-specific)
  - Context Window defines maximum tokens for consideration
  - Includes input prompt, conversation history, and output
  - Seemingly rebuilds conversation to maintain context

## Implementation Features

### Common Tool Applications
- Data/knowledge fetching
- Action execution (e.g., meeting booking)
- Calculation performance
- UI modification

### Agent Systems
- Autonomous task-performing entities
- Characteristics:
  - Autonomous, goal-oriented, specific
  - Framework integration for complex problem-solving with limited human involvement
  - Memory/persistence capabilities
  - Decision-making/orchestration
  - Planning abilities
  - Tool usage with database/internet connectivity

## Weekly Assignments

### Week 2: Speech-to-Text Options

#### Cloud Services
- Google Cloud Speech-to-Text API
- Amazon Transcribe
- Microsoft Azure Speech Services

#### Open Source Options
- OpenAI Whisper
- Web Speech API (for browser-based applications)

#### HuggingFace Platform
- 800,000 models
- 200,000 datasets
- Spaces with apps built in Gradio or other open source tools

### Week 3: Dataset Generation
- Create models that can create datasets
- Use enterprise and open source models to generate data
- Create Gradio UI for this product

### Week 4: Model Evaluation

#### Evaluation Criteria
- Open/closed source availability
- Release data (knowledge cutoff)
- Technical specifications:
  - Parameters
  - Training tokens
  - Context length
- Cost factors:
  - Inference cost (API charge, subscription or runtime compute)
  - Training cost
  - Build cost
- Performance metrics:
  - Time to market
  - Rate limits
  - Speed (how quickly a whole response is generated)
  - Latency (how quickly does it start responding)
- License requirements

#### Chinchilla Scaling Law
- Number of parameters is ~proportional to number of training tokens

#### 7 Common Benchmarks
- **ARC** → Reasoning → A benchmark for evaluating scientific reasoning
- **DROP** → Language Comprehension → Distill details from text then add, count, or sort
- **HellaSwag** → Common Sense → Harder endings, long contexts and low shot activities
- **MMLU** → Understanding → Factual recall, reasoning and problem solving across 57 subjects
- **TruthfulQA** → Accuracy → Robustness in providing truthful replies
- **Winograde** → Test the LLM understands context and resolves ambiguity
- **GSM8K** → Math → Math and word problems taught K-8

#### 3 Specialized Benchmarks
- **ELO** → Chat → Results from head-to-head face-offs with other LLMs (think Chess)
- **HumanEval** → Python coding → 164 problems writing code based on docstrings
- **MultiPL-E** → Broader Coding → Translation of HumanEval to 18 programming languages

#### Benchmark Limitations
- Not consistently applied, hard to measure
- Overfitting (trained for specific benchmark questions)
- Training data leakage
- Models may be aware they are being evaluated (unproven)

#### 6 New Benchmarks
- **GPQA (Google Proof Q&A)** → Graduate Tests → 448 expert questions; non-PhD humans score 34% even with web access (Claude 3.5 Sonnet is the best right now)
- **BBHard** → Future Capability → 204 tasks believed beyond capabilities of LLMs
- **Math Lv 5** → Math → High school level math competition problems
- **IfEval** → Difficult Instructions → "Write 400 words" and "mention AI at least 3 times"
- **MuSR** → Multistep Soft Reasoning → Logical deduction such as analyzing 1,000 word mystery and asking "motive, opportunity, means"
- **MMLU-PRO** → Harder MMLU → A more advanced and cleaned up version choice of 10 answers rather than 4

#### 6 Leaderboards
- HuggingFace Open LLM
- HuggingFace BigCode
- HF LLM Perf
- HF Others
- Vellum
- SEAL

#### LMSYS Chatbot Arena
- LLMs measured with ELO in head-to-head competition

### Week 5: Retrieval Augmented Generation
- Use a database of expert information (Knowledge Base)
- Every time user asks question, search for anything relevant in Knowledge Base
- Add relevant details in the prompt

#### Vector Embedding
- **Auto-regressive vs Auto-encoding LLMs**
  - Regressive: predict future token from the past
  - Encoding: produce output based on the full input
    - Applications include sentiment analysis and classification
    - Calculate vector embedding → take text and convert to series of numbers that reflect their meaning
    - Can represent character, word, token, entire document, something abstract
    - Typically hundreds of thousands of dimensions
    - Represent understanding of inputs
    - Support vector math "King - Man + Woman = Queen"
- **Technologies**
  - BERT from Google
  - OpenAIEmbeddings from OpenAI

#### LangChain Framework
- Declarative language: LangChain Expression Language (LCEL) used to interface with LLMs
- Simplifies creation of apps using LLMs - fast time to market
- Wrapper code around LLMs makes it easy to swap models
- APIs have matured, need for unifying framework like LangChain has decreased

### Week 6: Fine-tuning Frontier LLMs with LoRA/QLoRA

#### Inference Methods
- Trained LLM uses learned knowledge to generate outputs based on a given input
  - Multi-shot prompting
  - Prompt chaining
  - Tools/Function calling
  - RAG / Knowledge base

#### Transfer Learning
- Substitution for training which costs millions and billions of parameters
- Take pretrained model as base and use additional training

#### Finding Datasets
- Your own proprietary data
- Kaggle
- HuggingFace datasets
- Synthetic data (generated by LLM)
- Specialist companies like Scale.com

#### Data Processing
- Investigate
- Parse
- Visualize
- Assess Data Quality
- Curate
- Save

#### 5 Step Strategy

1. **Understand**
   - Gather requirements, identify performance criteria
   - Understand data (quantity, quality, format)
   - Non-functional requirements

2. **Prepare**
   - Research existing / non-LLM solutions
   - Compare relevant LLMs
   - Curate data: clean, preprocess, split

3. **Select Model**
   - Choose LLM
   - Experiment
   - Train and validate with curated data

4. **Customize**
   - **Prompting**: multi-shot, chaining and tools
     - Pros: fast to implement, low cost, often immediate improvement
     - Cons: limited context length, diminishing returns, slower and expensive inference (additional context)
   - **RAG**: accuracy improvement with low data needs, scalable, efficient
     - Cons: Harder to implement, requires up-to-date data, lacks nuance
   - **Fine-tuning**: additional training, deep expertise, nuance, learn different tone/style, faster and cheaper inference
     - Cons: Significant effort to implement, high data needs, training cost, risk of "catastrophic forgetting"

5. **Productionize**
   - Determine API between model and platform(s)
   - Identify model hosting and deployment architecture
   - Scaling, monitoring, security and compliance
   - Measure business performance metrics

#### Key Objectives of Fine-Tuning for Frontier Models
- Setting style or tone that cannot be achieved by prompting
- Improving reliability of producing a type of output
- Correcting failures to follow complex problems
- Handling edge cases
- Performing new skill or task that's hard to articulate in a prompt

## Model Engineering Techniques

### Quantization
- Reducing the precision of the weights in the model so that it will run faster and use less memory

### Model Internals
- PyTorch layers that sit behind HuggingFace library

### Streaming
- Streams of results that come back (text streaming)