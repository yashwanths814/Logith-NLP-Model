# 🧠 Logith NLP Model

A robust Natural Language Processing solution for classifying cybercrime complaints using Logistic Regression and TF-IDF. Developed for the **Cyberguard AI Hackathon**, this project addresses the urgent need for scalable text analytics in the face of rising cybercrime reports.

## 📌 Problem Statement

With cybercrime complaints surging from 26,049 in 2019 to over 1.5 million in 2023, the National Cyber Crime Reporting Portal (NCRP) faces a data overload. This model aims to automate complaint categorization into:
- 👩‍👧 Women/Child Related Crimes
- 💸 Financial Fraud Crimes
- 🕵️ Other Cyber Crimes

## 🎯 Objectives

- Perform EDA to uncover patterns in complaint text
- Clean and preprocess raw data for modeling
- Train a Logistic Regression classifier using TF-IDF features
- Evaluate model performance using standard metrics

## 🧪 Dataset Overview

- `train_Dataset.csv` and `test_Dataset.csv`
- Key columns: `complaint_text`, `crimeaditionalinfo`, `category`

## 🛠️ Preprocessing Pipeline

1. Lowercasing
2. Removing special characters and digits
3. Tokenization
4. Stopword removal (NLTK)
5. Stemming (Porter Stemmer)
6. TF-IDF vectorization (max 1000 features)

## 🧬 Model Architecture

- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF
- **Split**: 80/20 train-test
- **Metrics**: Accuracy, Precision, Recall, F1 Score


## 🚀 How to Run
# Step 1: Clone the repository
git clone https://github.com/yashwanths814/Logith-NLP-Model

# Step 2: Navigate to the project folder
cd Logith-NLP-Model

# Step 3: Install dependencies
pip install pandas numpy scikit-learn nltk

# Step 4: Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

