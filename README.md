
# **Aspect-Based Sentiment Analysis (CITS4012, UWA)**  

This project was part of the **NLP course (CITS4012) at UWA** and achieved the **highest marks in the batch**. The code implements **Aspect-Based Sentiment Analysis (ABSA)** using deep learning techniques.

---

## **Project Overview**  

The code focuses on classifying sentiment (positive, negative, neutral) for specific aspects within a sentence. Several deep learning architectures were explored, including:  

- **BiLSTM with Aspect Integration**  
- **BiLSTM + POS Tags**  
- **BiLSTM + Additive Attention**  
- **BiLSTM + Multi-head Attention**  
- **Skip-Gram-based Word Embeddings**  

The models were trained and evaluated on a structured dataset consisting of sentences labeled with aspects and sentiments. Various techniques like N-grams, aspect embeddings, and attention mechanisms were tested for performance improvements.

---

## **Code Explanation**  

### **1️⃣ Data Processing**  
- Loads dataset from JSON files (`train.json`, `val.json`, `test.json`).  
- Cleans text using **tokenization, lemmatization, and stopword removal**.  
- Implements **aspect integration** and **N-grams generation**.  

### **2️⃣ Model Implementations**  
- **Model 1**: Uses **BiLSTM** with aspect-aware embeddings.  
- **Model 2**: Introduces **POS tags** for additional contextual understanding.  
- **Model 3**: Implements **attention mechanisms**:
  - **(a) Additive Attention**
  - **(b) Multi-head Attention**
  - **(c) Aspect-aware Attention  

### **3️⃣ Evaluation & Visualization**  
- **Confusion matrices** for model predictions.  
- **Ablation studies** comparing different architectures.  
- **Hyperparameter tuning** (optimizers, batch sizes).  

---

## **Running the Code**  

Ensure **Python** and the required libraries are installed. The models rely on **TensorFlow, Keras, NLTK, and SpaCy**.  

### **Install Dependencies**  
```bash
pip install tensorflow keras nltk spacy matplotlib scikit-learn gensim
python -m spacy download en_core_web_sm
```

### **Train a Model**  
Modify the script to load data and train the desired model. Example:  
```python
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=64)
```

### **Evaluate Performance**  
```python
model.evaluate(test_data, test_labels)
```

### **Plot Results**  
Confusion matrix example:  
```python
plot_confusion_matrix(model, "Model Name", test_data, test_labels)
```
