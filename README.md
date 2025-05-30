# Custom Neural Network Chatbot for Market Research

**A Learning Journey: Building Custom Neural Network Architecture from Scratch**

This project represents my hands-on exploration of creating a custom neural network chatbot using my own database. Built entirely from scratch using TensorFlow/Keras, this chatbot is trained on a proprietary dataset of 751 market research Q&A pairs and demonstrates the complete process of designing, implementing, and training a domain-specific conversational AI system.

## Project Overview

**Learning Objective**: Understanding how to create your own custom neural network architecture using your own database that acts as a chatbot.

This project was developed as an educational deep-dive into the fundamentals of neural network architecture design for conversational AI. Rather than relying on pre-built models or APIs, I built this chatbot from the ground up to gain comprehensive understanding of:

- Custom neural network architecture design
- Sequence-to-sequence modeling for conversation
- Text preprocessing and tokenization
- Vocabulary building and word embeddings
- Training a chatbot on domain-specific data

## Neural Network Architecture

The chatbot implements a **Bidirectional LSTM Encoder-Decoder architecture** with the following layers:

### Architecture Components:

1. **Embedding Layer**
   - Converts word indices to dense vectors using GloVe embeddings (100d)
   - Vocabulary size: Built from training data with minimum frequency filtering

2. **Encoder (Bidirectional LSTM)**
   - Bidirectional LSTM with 256 units each direction (512 total)
   - Processes input questions and generates context representations
   - Dropout: 0.3 for regularization

3. **Decoder (LSTM)**
   - LSTM with 512 units for response generation
   - Takes encoder states as initial states
   - Dropout: 0.3 for regularization

4. **Attention Mechanism**
   - Dense layer with tanh activation
   - Helps focus on relevant parts of input during generation

5. **Output Layer**
   - Dense layer with softmax activation
   - Generates probability distribution over vocabulary

### Model Statistics:
- **Training Samples**: 750 Q&A pairs
- **Vocabulary Size**: Dynamic (built from training data)
- **Parameters**: Variable based on vocabulary size
- **Architecture**: Custom seq2seq with attention

## Dataset

The model is trained on a custom dataset containing 751 market research question-answer pairs covering:
- Market sizing and analysis
- Startup and business strategies
- Industry trends and insights
- Financial metrics and KPIs
- Business validation techniques

**Dataset Structure:**
```
Question,Answer
"What is the market size of edtech?","The global edtech market was valued at..."
"How to validate a business idea?","Business idea validation involves..."
```

### Key Features:
- **Text Preprocessing**: Custom tokenization, lemmatization, and stopword removal
- **Vocabulary Building**: Dynamic vocabulary creation with frequency filtering
- **GloVe Embeddings**: Pre-trained word embeddings for better semantic understanding
- **Hybrid Response System**: Combines similarity search with neural generation
- **Model Persistence**: Complete save/load functionality for deployment

### Response Generation Strategy:
1. **Similarity Search**: First attempts to find similar questions using token overlap
2. **Neural Generation**: Falls back to seq2seq model for novel responses
3. **Fallback Handling**: Graceful handling of out-of-domain queries

### Training from Scratch
1. Open `MarketResearchChatbot.ipynb` in Jupyter Notebook/Google Colab
2. Ensure `market_research_chatbot_dataset.csv` is in the same directory
3. Run all cells to train your custom model
4. Models will be automatically saved for future use

## Learning Objectives Achieved

Through this project, I gained hands-on experience with:

- **Neural Architecture Design**: Building custom seq2seq models from scratch
- **NLP Preprocessing**: Text cleaning, tokenization, and vocabulary management
- **Embedding Integration**: Using pre-trained GloVe embeddings effectively
- **Training Optimization**: Implementing callbacks, learning rate scheduling
- **Model Persistence**: Saving and loading complex multi-model systems
- **Inference Pipeline**: Building complete end-to-end conversation systems

## Customization

To adapt this chatbot for your domain:

1. **Replace Dataset**: Update with your Q&A pairs in CSV format
2. **Adjust Parameters**: Modify vocabulary size, sequence lengths, embedding dimensions
3. **Fine-tune Architecture**: Experiment with different LSTM sizes, attention mechanisms
4. **Domain-specific Preprocessing**: Add custom text cleaning for your use case

##  Performance Considerations

### Optimization Techniques Used:
- **Early Stopping**: Prevents overfitting during training
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Dropout Regularization**: Reduces overfitting in LSTM layers
- **Gradient Clipping**: Implicitly handled by Adam optimizer
- **Batch Processing**: Efficient training with appropriate batch sizes

### Response Quality:
- Combines similarity matching with neural generation
- Fallback mechanisms for out-of-domain queries
- Context-aware responses through attention mechanisms


**Core Learning Focus**: How to create your own custom neural network architecture using your own database that acts as a chatbot.

This project serves as a comprehensive example of:
- **Building neural networks from first principles** - No pre-built chatbot APIs used
- **Custom architecture design** - Every layer designed and implemented manually  
- **Database-driven training** - Using my own curated dataset of 751 Q&A pairs
- **Understanding seq2seq architectures deeply** - From theory to implementation
- **Implementing attention mechanisms manually** - Built attention layer from scratch
- **Creating domain-specific AI systems** - Tailored for market research domain
- **Complete ML pipeline** - From raw data to deployable chatbot

**Key Learning Achievement**: Gained deep understanding of how chatbots work internally by building one completely from scratch using custom neural network architecture and personal database.
