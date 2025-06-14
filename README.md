# 💬 Sentiment Analysis on Movie Reviews

A machine learning project that classifies movie reviews as **positive** or **negative** using Natural Language Processing (NLP) techniques. Trained on the **IMDB dataset** using TF-IDF and a Logistic Regression classifier.

---

## 🔍 Objective

To build a model that automatically detects the **sentiment** of a given movie review — helping users understand general audience opinion.

---

## 🚀 Features

- Preprocessing of raw text data (cleaning, tokenization, stopword removal)
- Vectorization using **TF-IDF**
- Model training using:
  - Logistic Regression
  - Multinomial Naive Bayes (optional)
- Evaluation using **accuracy, precision, recall, F1-score**
- Simple command-line or notebook-based interface for predictions

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK / SpaCy (for NLP preprocessing)
- Matplotlib / Seaborn (for visualization)

---

## 📁 Dataset

- [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
  Contains 50,000 labeled movie reviews (25k train, 25k test)

---

## 🧪 How It Works

1. **Data Preprocessing**  
   Clean reviews → remove punctuation, lowercasing, stopwords, lemmatization.

2. **TF-IDF Vectorization**  
   Convert text into numerical vectors representing importance of words.

3. **Model Training**  
   Use Logistic Regression to classify reviews into **positive** or **negative**.

4. **Model Evaluation**  
   Metrics: Accuracy, Confusion Matrix, ROC Curve, etc.

---


