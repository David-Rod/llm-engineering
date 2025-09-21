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

#### LoRA (Low-Rank Adaptation)
- **Definition**: A parameter-efficient fine-tuning technique that reduces the number of trainable parameters by orders of magnitude while maintaining model performance
- **Key Concept**: Instead of updating all model weights, LoRA freezes the original model weights and introduces trainable rank decomposition matrices alongside existing weights --> Smaller matrices with fewer dimensions and train those new matrices. These are based on target module in the data
- **Technical Details**:
  - Decomposes weight updates into two smaller matrices (A and B) where the rank is much smaller than the original weight matrix
  - Original weight W₀ remains frozen
- **Benefits**:
  - Reduces trainable parameters by 10,000x (e.g., GPT-3 175B → 4.7M trainable parameters)
  - Maintains model quality comparable to full fine-tuning
  - Enables fine-tuning on consumer hardware
  - Multiple LoRA adapters can be trained for different tasks and swapped efficiently

#### QLoRA (Quantized LoRA)
- **Definition**: An extension of LoRA that combines quantization with low-rank adaptation for even more memory-efficient fine-tuning, keep weights but reduce their precision by reducing memory needed to store weight value (32 bit to 8 bit or 4 bit)
- **Key Innovation**: Uses 4-bit quantization of the base model while keeping LoRA adapters in higher precision
- **Technical Components**:
  - **4-bit NormalFloat (NF4)**: Information-theoretically optimal quantization scheme for normally distributed weights
  - **Double Quantization**: Quantizes the quantization constants themselves to save additional memory
  - **Paged Optimizers**: Uses NVIDIA unified memory to handle memory spikes during training
- **Memory Efficiency**:
  - Reduces memory usage by ~65% compared to LoRA
  - Enables fine-tuning of 65B parameter models on a single 48GB GPU
  - Maintains performance within 1% of full 16-bit fine-tuning
- **Workflow**:
  1. Load pre-trained model in 4-bit precision
  2. Add LoRA adapters in 16-bit precision
  3. Train only the LoRA parameters while keeping base model frozen and quantized

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

## Hyperparameters in LLM Training

A **hyperparameter** in the context of training LLMs is a configuration setting that you choose before training begins and that controls how the training process works, rather than being learned by the model itself.

### Key Characteristics:
- **Set by humans**, not learned by the model
- **Controls the training process** and model architecture
- **Requires experimentation** to find optimal values
- **Affects model performance** but isn't part of the learned weights

### Common LLM Hyperparameters:

#### Training Process
- **Learning rate**: How fast the model updates its weights
- **Batch size**: Number of examples processed together
- **Number of epochs**: How many times to go through the entire dataset
- **Optimizer type**: Algorithm used to update weights (Adam, SGD, etc.)

#### Model Architecture
- **Number of layers**: Depth of the neural network
- **Hidden dimensions**: Size of internal representations
- **Attention heads**: Number of parallel attention mechanisms
- **Context length**: Maximum sequence length the model can process

#### Regularization
- **Dropout rate**: Percentage of neurons randomly turned off during training
- **Weight decay**: Penalty for large weights to prevent overfitting

#### Example from Course:
The **180 token limit** for product descriptions is a hyperparameter choice - determined through experimentation to balance having enough information for pricing while keeping training efficient. The instructor noted this was "trial and error" to find the right balance between information richness and computational efficiency - which is exactly how


### Week 7: Fine-tuning Open-Source LLMs with LoRA/QLoRA
#### 3 Hyperparameters pertinent ot QLoRA
- **R**: Rank, how many dimensions in the low-rank matrices (Start with 8)
- **Alpha**: Scaling factor the multiplies the lower rank matrices (Aplha * A matrix * B matrix), Start with 2x value of R
- **Target Modules**: which layers in neural network are adapted, target the attention head layers


### LLM Hyperparameter Definitions

#### Core Hyperparameters

##### Epochs
Complete passes through the entire training dataset. One epoch = the model has seen every training example once. More epochs can improve learning but risk overfitting.

##### Batch Size
Number of training examples processed together before updating model weights. Larger batches provide more stable gradient estimates but require more memory.

##### Learning Rate
Step size for weight updates during training. Controls how much the model changes with each update. Too high causes instability; too low causes slow learning.

##### Gradient Accumulation
Technique to simulate larger batch sizes by accumulating gradients over multiple smaller batches before updating weights. Useful when memory limits prevent large batches.

##### Optimizer
Algorithm that determines how to update model weights based on gradients. Common types include Adam, SGD, and AdamW. Different optimizers have different convergence properties and memory requirements.

#### Key Relationships

##### Batch Size ↔ Gradient Accumulation
Inversely related - use gradient accumulation when you can't fit desired batch size in memory

##### Learning Rate ↔ Batch Size
Larger batches often allow higher learning rates (more stable gradients)

##### Learning Rate ↔ Optimizer
Different optimizers respond differently to learning rate changes; Adam typically needs lower rates than SGD

##### Epochs ↔ Learning Rate
Often decay learning rate over epochs to fine-tune convergence

##### Training Time Considerations
All parameters affect training time: Larger batches and higher learning rates can reduce epochs needed, but may hurt final performance


#### Training Process (4 steps)
- Forward pass - predict the next token in training data
- Loss calculation - How different was the predicted token to the actual next token
- Backward pass (backward propogation) - How much should the parameters be tweaked to enhance performance
- Optimization - updates parameters a tiny step to do better (weight adjustments)
