# Open Source Gen AI and LLM Engineering

## HuggingFace Platform
- **800,000 models**
- **200,000 datasets**
- **Spaces**
  - Apps built in gradio or other open source tools

## Model Engineering Techniques
- **Quantization**
  - Reducing weight precision for faster runtime and lower memory usage
- **Model Internals**
  - Pytorch layers behind hugging face library
- **Streaming**
  - Text streaming response capabilities

## Weekly Projects

### Week 3: Dataset Generation
- Create models that generate datasets
- Use enterprise and open source models
- Create Gradio UI for the product

### Week 4: Model Evaluation Criteria
- **Deployment Options**
  - Open/closed source
  - Release data (knowledge cutoff)
- **Model Characteristics**
  - Parameters
  - Training tokens
  - Context length
- **Cost Factors**
  - Inference cost (API, subscription, runtime)
  - Training cost
  - Build cost
- **Performance Metrics**
  - Time to market
  - Rate limits
  - Speed (full response generation)
  - Latency (initial response time)
- **Legal**
  - License requirements

#### Chinchilla Scaling Law
- Number of parameters ≈ number of training tokens

### Model Benchmarks

#### Standard Benchmarks
1. **ARC**: Scientific reasoning
2. **DROP**: Language comprehension
3. **HellaSwag**: Common sense
4. **MMLU**: Multi-subject understanding
5. **TruthfulQA**: Response accuracy
6. **Winograde**: Context understanding
7. **GSM8K**: K-8 mathematics

#### Specialized Benchmarks
1. **ELO**: Chat comparison
2. **HumanEval**: Python coding
3. **MultiPL-E**: Multi-language coding

#### New Benchmarks
1. **GPQA**: Graduate-level testing
2. **BBHard**: Future capabilities
3. **Math Lv 5**: Competition math
4. **IfEval**: Complex instructions
5. **MuSR**: Multi-step reasoning
6. **MMLU-PRO**: Advanced MMLU

#### Leaderboards
- HuggingFace (Open LLM, BigCode, LLM Perf, Others)
- Vellum
- SEAL
- LMSYS Chatbot Arena (ELO rankings)

### Week 5: RAG (Retrieval Augmented Generation)
- Knowledge Base integration
- Contextual search
- Prompt enhancement

#### Vector Embedding
- **LLM Types**
  - Auto-regressive: Future token prediction
  - Auto-encoding: Full input processing
    - Applications: Sentiment analysis, classification
    - Vector embedding capabilities
    - High-dimensional representations
    - Semantic mathematics
- **Technologies**
  - BERT (Google)
  - OpenAIEmbeddings

#### LangChain Framework
- Declarative LCEL
- Rapid development
- Model abstraction
- Decreasing necessity with API maturity

### Week 6: Fine-tuning Frontier LLMs

#### Inference Methods
- Multi-shot prompting
- Prompt chaining
- Tools/Function calling
- RAG integration

#### Transfer Learning
- Base model adaptation
- Dataset sources:
  - Proprietary data
  - Kaggle
  - HuggingFace
  - Synthetic data
  - Specialist providers

#### Implementation Strategy
1. **Understand**
   - Requirements gathering
   - Performance criteria
   - Data assessment

2. **Prepare**
   - Solution research
   - LLM comparison
   - Data curation

3. **Select Model**
   - LLM selection
   - Experimentation
   - Training/validation

4. **Customize**
   - Prompting techniques
   - RAG implementation
   - Fine-tuning process

5. **Productionize**
   - API design
   - Deployment architecture
   - Operations management
   - Performance monitoring

#### Fine-Tuning Objectives
- Style/tone customization
- Output reliability
- Complex instruction handling
- Edge case management
- New