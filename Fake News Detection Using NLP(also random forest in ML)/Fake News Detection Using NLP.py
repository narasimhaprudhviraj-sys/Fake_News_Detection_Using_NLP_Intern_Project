# ==========================================
# FAKE NEWS DETECTION USING NLP
# RANDOM FOREST CLASSIFIER
# ==========================================

import pandas as pd
import numpy as np
import os
import re
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# DOWNLOAD STOPWORDS
# ==========================================
nltk.download('stopwords')

# ==========================================
# LOAD DATASETS
# ==========================================
print("Loading Dataset...")
current_dir = os.path.dirname(os.path.abspath(__file__))

fake_path = os.path.join(current_dir, "Fake_200.csv")
true_path = os.path.join(current_dir, "True_200.csv")

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)
print("\nDataset Loaded Successfully")
# ==========================================
# REDUCE DATASET TO 400 RECORDS
# ==========================================
fake_small = fake.sample(n=200, random_state=42)
true_small = true.sample(n=200, random_state=42)

# Add Labels
fake_small["label"] = 0   # Fake News
true_small["label"] = 1   # Real News

# Merge Data
data = pd.concat([fake_small, true_small])

# Shuffle Data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save Reduced Dataset
data.to_csv("fake_news_400.csv", index=False)

print("\nDataset Created Successfully")
print("Total Records:", len(data))

# ==========================================
# NLP TEXT CLEANING
# ==========================================
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'https?://\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

print("\nCleaning Text Data...")

data["text"] = data["text"].apply(clean_text)

# ==========================================
# FEATURES AND LABELS
# ==========================================
X = data["text"]
y = data["label"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

# ==========================================
# TF-IDF FEATURE EXTRACTION
# ==========================================
vectorizer = TfidfVectorizer(max_features=3000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================
print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_tfidf, y_train)

# ==========================================
# PREDICTION
# ==========================================
y_pred = rf_model.predict(X_test_tfidf)

# ==========================================
# MODEL EVALUATION
# ==========================================
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot()

plt.title("Fake News Detection Confusion Matrix")
plt.show()

# ==========================================
# USER INPUT PREDICTION WITH REPORT
# ==========================================

def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    vector = vectorizer.transform([cleaned_text])

    prediction = rf_model.predict(vector)

    probability = rf_model.predict_proba(vector)

    fake_score = probability[0][0] * 100
    real_score = probability[0][1] * 100

    print("\n===================================")
    print("      FAKE NEWS ANALYSIS REPORT")
    print("===================================")

    print("\nInput News:")
    print(news_text)

    if prediction[0] == 0:
        print("\nResult : FAKE NEWS")
        print(f"Confidence : {fake_score:.2f}%")
    else:
        print("\nResult : REAL NEWS")
        print(f"Confidence : {real_score:.2f}%")

    print("\nDetailed Scores:")
    print(f"Fake News Probability : {fake_score:.2f}%")
    print(f"Real News Probability : {real_score:.2f}%")

    print("\n===================================")

# ==========================================
# TEST WITH USER INPUT
# ==========================================

while True:

    print("\nEnter a News Headline or Article")
    news = input("\nNews: ")

    predict_news(news)

    choice = input("\nCheck another news? (yes/no): ")

    if choice.lower() != "yes":
        break

print("\nProject Completed Successfully!")