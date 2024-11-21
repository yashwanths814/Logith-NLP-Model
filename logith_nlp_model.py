
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Sample Data Loading (replace this with your actual dataset)
# Assuming 'complaint_text' is the text column and 'label' is the target variable
data = pd.read_csv('train.csv')  # Load your dataset
data.dropna(inplace=True)  # Drop missing values for simplicity

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization and stopword removal
    tokens = [word for word in text.split() if word not in stop_words]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_text'] = data['crimeaditionalinfo'].apply(preprocess_text)

# Splitting data into features and labels
X = data['cleaned_text']
y = data['category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features based on dataset size
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Development: Using Naive Bayes as an example
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predicting and Evaluating the Model
y_pred = model.predict(X_test_vec)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Displaying Results
print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")