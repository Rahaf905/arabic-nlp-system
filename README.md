# Arabic NLP System
A comprehensive Natural Language Processing project for Arabic text, covering text preprocessing, classification, machine translation, and question answering using traditional machine learning, deep learning, and transformer-based models.

## Project Overview
This project explores multiple Arabic NLP tasks and compares different text representation techniques and learning architectures. It includes experiments with traditional feature extraction methods, static and contextual embeddings, classical machine learning classifiers, Seq2Seq networks, and transformer-based generative models.

## Main Tasks
### 1. Arabic Text Preprocessing
The preprocessing pipeline includes:
- Text cleaning and normalization
- Tokenization
- Stopword removal
- Arabic stemming
- Handling Arabic linguistic variations
- Preparing text for classification and generation tasks

### 2. Text Classification
Arabic questions were classified using multiple text representations and machine learning models.

#### Text Representations
- Bag of Words
- TF-IDF
- Word2Vec
- FastText
- AraBERT embeddings
- Qwen embeddings
- Multilingual E5 embeddings
- BGE-M3 embeddings

#### Classification Models
- Support Vector Machine
- Logistic Regression
- Random Forest
- Naive Bayes
- Fine-tuned AraBERT
The best classification performance was achieved using Qwen contextual embeddings with an SVM classifier, reaching 78.4% accuracy.

### 3. Arabic-to-English Machine Translation
Arabic text was translated into English using GPT-4o-mini through an API-based translation pipeline.
The API key is intentionally excluded from this repository for security.

### 4. Arabic Question Answering
The question-answering task was investigated using both Seq2Seq and transformer-based models.

#### Seq2Seq Architectures
- Simple RNN
- LSTM
- GRU

These architectures were evaluated with different embeddings, including FastText, BERT, Qwen, E5, and BGE.
FastText combined with GRU achieved the best validation accuracy among the Seq2Seq experiments.

#### Transformer-Based Models
- mT5
- Qwen
- AraGPT2

The models were evaluated using validation loss and BLEU score. mT5 achieved the highest BLEU score among the transformer-based question-answering models.

## Interface
The project includes an interactive interface that allows users to experiment with the implemented NLP tasks and models.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- PyArabic
- Scikit-learn
- TensorFlow
- PyTorch
- Hugging Face Transformers
- Gensim
- FastText
- Streamlit
- OpenAI API
- Google Colab

## Project Structure
- `app/` — Interactive application interface
- `notebooks/` — Preprocessing, classification, translation, Seq2Seq, and transformer experiments
- `presentation/` — Project presentation and results analysis

## Main Results
- Qwen embeddings with SVM achieved the best classification accuracy.
- Contextual embeddings generally outperformed static word representations.
- TF-IDF with stemming achieved the best result among traditional representations.
- FastText with GRU achieved the highest Seq2Seq validation accuracy.
- mT5 achieved the highest BLEU score among the transformer-based QA models.

## Dataset
The project uses an Arabic question-and-answer dataset containing approximately 5,000 samples. The dataset and generated embedding files are not included in this repository.

## Author

Rahaf Al-Shami  
Natural Language Processing Course Project
