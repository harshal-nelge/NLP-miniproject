# News Classification Project

## Course: NLP (Semester 6) - Pillai College of Engineering

### Project Overview
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The project focuses on **News Classification**, where we apply various **Machine Learning (ML), Deep Learning (DL), and Language Models** to categorize news articles into predefined categories. This project involves exploring techniques like **text preprocessing, feature extraction, model training, and evaluating models** for their effectiveness in classifying news articles.

🔗 You can learn more about the college by visiting the [official website of Pillai College of Engineering](https://www.pce.ac.in).

---

## 🎓 Acknowledgements
We would like to express our sincere gratitude to the following individuals:

### 📖 Theory Faculty  
- **Dhiraj Amin**  
- **Sharvari Govilkar**  

### 💻 Lab Faculty  
- **Dhiraj Amin**  
- **Neha Ashok**  
- **Shubhangi Chavan**  

Their guidance and support have been invaluable throughout this project.

---

## 📰 News Classification using Natural Language Processing

### Project Abstract
The **News Classification** project aims to classify news articles into different categories like **Sports, Politics, Technology, Entertainment, and Business**. This task involves applying **Machine Learning, Deep Learning, and Language Models** to accurately categorize news text based on its content. The project explores different algorithms, including traditional machine learning techniques, deep learning models, and state-of-the-art pre-trained language models. The goal is to **evaluate the performance of each approach and select the best-performing model** for news classification.

---

## **Algorithms Used**

### **Machine Learning Algorithms**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

### **Deep Learning Algorithms**
- **Convolutional Neural Networks (CNN)**
- **Recurrent Neural Networks (RNN)**
- **Long Short-Term Memory (LSTM)**

### **Language Models**
- **GPT**
- **BERT** (Bidirectional Encoder Representations from Transformers)

---

## **Comparative Analysis**
The comparative analysis of different models highlights their effectiveness in classifying news articles into the correct category. Below are the summarized performance metrics for various models tested:

| **Model Type**           | **Accuracy (%)** | **Precision (%)** | **Recall (%)** | **F1-Score (%)** |
|--------------------------|------------------|------------------|--------------|--------------|
| Logistic Regression      | 82.5             | 81.3             | 84.2         | 82.7         |
| SVM (Support Vector Machine) | 85.3        | 83.6             | 87.4         | 85.5         |
| Random Forest           | 88.1             | 86.7             | 89.2         | 87.9         |
| CNN (Convolutional Neural Networks) | 91.2  | 90.0             | 92.4         | 91.2         |
| RNN (Recurrent Neural Networks) | 89.5    | 88.0             | 91.0         | 89.5         |
| LSTM (Long Short-Term Memory) | 92.0      | 91.1             | 93.0         | 92.0         |
| **BERT**                | **94.5**         | **93.8**         | **95.2**     | **94.5**     |

---

## **Conclusion**
This **News Classification** project demonstrates the potential of **Machine Learning, Deep Learning, and Language Models** for text classification tasks, particularly for categorizing news articles. The comparative analysis reveals that **BERT**, a transformer-based model, outperforms traditional methods and deep learning models in terms of **accuracy, precision, and recall**. By employing various algorithms, we gain insights into the **strengths and weaknesses of each model**, allowing us to choose the most suitable approach for **news classification**.

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AddyDev555/News_Classification.git
   cd News_Classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pretrained models (if applicable):
   ```bash
   python download_models.py
   ```

---

## 🛠️ Usage

### 1️⃣ **Preprocess News Articles**
```python
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded_sequence
```

### 2️⃣ **Evaluate with CNN Model**
```python
model = load_model("news_classification_cnn.h5")
def predict_category(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    predicted_category = np.argmax(prediction, axis=1)[0]
    return predicted_category
```

### 3️⃣ **Classify News Articles using BERT**
```python
def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predicted_category = torch.argmax(outputs.logits, dim=1).item()
    return predicted_category
```

---

## 📈 Results & Insights
- **BERT achieves the highest accuracy (94.5%)**, making it the best choice for news classification.
- **LSTM and CNN** models also show strong performance, with accuracies of **92.0% and 91.2%** respectively.
- **Traditional ML models like Random Forest and SVM** perform well but are outclassed by deep learning models.

---

## 🤝 Contributing
We welcome contributions! To contribute:
1. **Fork the repository**
2. **Open an issue** to discuss your proposed changes.
3. **Submit a pul
