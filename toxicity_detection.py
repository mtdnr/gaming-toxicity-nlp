import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# LOAD DATASET
# ============================================
print("Loading dataset...")
# Using header=1 because your CSV has an empty row at the top!
df = pd.read_csv("Toxicity Dataset.csv", header=1) 

print("\nDataset Preview:")
print(df.head())

# ============================================
# HANDLE MISSING VALUES
# ============================================
df["Previous_Chat"] = df["Previous_Chat"].fillna("[start of chat]")
df["Current_Message"] = df["Current_Message"].fillna("")

# ============================================
# COMBINE CONTEXT + CURRENT MESSAGE
# ============================================
df["Combined_Text"] = df["Previous_Chat"] + " " + df["Current_Message"]

# ============================================
# TEXT CLEANING FUNCTION
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) # Remove URLs
    text = re.sub(r"[^a-z0-9\s!?']", " ", text) # Keep basic punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text

# Apply cleaning
df["Cleaned_Text"] = df["Combined_Text"].apply(clean_text)

X = df["Cleaned_Text"]
y = df["Label"]

# ============================================
# TRAIN / TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# ============================================
# BASELINE MODEL - NAIVE BAYES
# ============================================
print("\nTraining Baseline Model (Naive Bayes)...")
baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("classifier", MultinomialNB())
])
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

# ============================================
# IMPROVED MODEL - LOGISTIC REGRESSION
# ============================================
print("Training Improved Model (Logistic Regression)...")
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ============================================
# EVALUATION & TEXT OUTPUT
# ============================================
print("\n===================================")
print("MODEL EVALUATION")
print("===================================")
print(f"Baseline (Naive Bayes) Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"Improved (Logistic Regression) Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred))

# ============================================
# 📊 GENERATE VISUAL GRAPHS 📊
# ============================================
print("\nGenerating graphs... (Please close the graph windows to continue to real-time testing)")

# 1. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Toxic (0)', 'Toxic (1)'], 
            yticklabels=['Non-Toxic (0)', 'Toxic (1)'],
            annot_kws={"size": 16})
plt.title('Confusion Matrix - Logistic Regression', fontsize=14)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300) # Saves the image to your folder
plt.show() # Displays the image on your screen! CLOSE IT TO CONTINUE.

# 2. Model Comparison Bar Chart
models = ['Naive Bayes\n(Baseline)', 'Logistic Regression\n(Improved)']
accuracies = [baseline_accuracy * 100, accuracy * 100]

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=models, y=accuracies, palette=['#cccccc', '#2ca02c'], hue=models, legend=False)
plt.title('Model Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)

# Add the percentage numbers on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f') + '%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=300) # Saves the image to your folder
plt.show() # Displays the image on your screen! CLOSE IT TO CONTINUE.

# ============================================
# REAL-TIME TOXICITY DETECTION
# ============================================
label_mapping = {0: "Non-Toxic", 1: "Toxic"}

print("\n===================================")
print("REAL-TIME TOXICITY DETECTION")
print("===================================")

while True:
    user_input = input("\nEnter a gaming chat message (or type 'exit'): ")
    if user_input.lower() == "exit":
        print("\nProgram terminated.")
        break

    cleaned_input = clean_text(user_input)
    proba_non_toxic, proba_toxic = model.predict_proba([cleaned_input])[0]
    threshold = 0.50
    prediction = 1 if proba_toxic >= threshold else 0
    toxic_score = proba_toxic * 100

    print("\nMessage:", user_input)
    print("Prediction:", label_mapping[prediction])
    print(f"Non-Toxic prob: {proba_non_toxic:.2f}, Toxic prob: {proba_toxic:.2f}")
    print(f"Confidence Score: {toxic_score:.2f}%")