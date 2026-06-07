import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import font

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
# 1. LOAD DATASET & PREPROCESS
# ============================================
print("Loading dataset and training models... (Please wait)")
df = pd.read_csv("Toxicity Dataset.csv", header=1) 
df["Previous_Chat"] = df["Previous_Chat"].fillna("[start of chat]")
df["Current_Message"] = df["Current_Message"].fillna("")
df["Combined_Text"] = df["Previous_Chat"] + " " + df["Current_Message"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-z0-9\s!?']", " ", text) 
    text = re.sub(r"\s+", " ", text).strip() 
    return text

df["Cleaned_Text"] = df["Combined_Text"].apply(clean_text)

X = df["Cleaned_Text"]
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================
# 2. TRAIN MODELS
# ============================================
baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("classifier", MultinomialNB())
])
baseline_model.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training Complete! LR Accuracy: {accuracy * 100:.2f}%")

# ============================================
# 3. GENERATE VISUAL GRAPHS (Optional: comment out if you just want the GUI)
# ============================================
print("\nGenerating graphs... (Close the image windows to launch the GUI App)")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('confusion_matrix.png', dpi=300) 
plt.show() 

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=['Naive Bayes', 'Logistic Regression'], y=[baseline_accuracy * 100, accuracy * 100], palette=['#cccccc', '#2ca02c'])
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f') + '%', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
plt.savefig('model_comparison_chart.png', dpi=300)
plt.show()

# ============================================
# 4. GAMER DARK MODE GUI
# ============================================
def analyze_message():
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input:
        return

    cleaned_input = clean_text(user_input)
    proba_non_toxic, proba_toxic = model.predict_proba([cleaned_input])[0]
    
    toxic_score = proba_toxic * 100
    
    # Update UI based on prediction
    if proba_toxic >= 0.50:
        result_label.config(text="⚠️ TOXIC DETECTED", fg="#ff4d4d") # Red
        bg_frame.config(bg="#331414") # Dark red warning background
    else:
        result_label.config(text="✅ SAFE MESSAGE", fg="#00cc66") # Green
        bg_frame.config(bg="#1a1a1a") # Normal dark background
        
    details_label.config(text=f"Confidence: {toxic_score:.1f}% Toxic\n\nProbability Breakdown:\nToxic: {proba_toxic:.2f} | Safe: {proba_non_toxic:.2f}")

# Initialize Window
root = tk.Tk()
root.title("FYP: Gaming Toxicity Detector")
root.geometry("550x450")
root.configure(bg="#1a1a1a") # Dark background

# Custom Fonts
title_font = font.Font(family="Helvetica", size=16, weight="bold")
main_font = font.Font(family="Helvetica", size=12)

# UI Elements
bg_frame = tk.Frame(root, bg="#1a1a1a", padx=30, pady=30)
bg_frame.pack(expand=True, fill=tk.BOTH)

header_label = tk.Label(bg_frame, text="Live Chat Moderator API", font=title_font, bg="#1a1a1a", fg="#ffffff")
header_label.pack(pady=(0, 20))

text_entry = tk.Text(bg_frame, height=4, width=50, font=main_font, bg="#2b2b2b", fg="#ffffff", insertbackground="white", relief=tk.FLAT)
text_entry.pack(pady=10)
text_entry.insert(tk.END, "Type a gaming message here...")

analyze_btn = tk.Button(bg_frame, text="SCAN MESSAGE", font=title_font, bg="#4da6ff", fg="#ffffff", activebackground="#3380cc", activeforeground="white", relief=tk.FLAT, command=analyze_message, cursor="hand2")
analyze_btn.pack(pady=20, fill=tk.X)

result_label = tk.Label(bg_frame, text="Awaiting Input...", font=title_font, bg="#1a1a1a", fg="#8c8c8c")
result_label.pack(pady=(10, 5))

details_label = tk.Label(bg_frame, text="", font=main_font, bg="#1a1a1a", fg="#cccccc")
details_label.pack()

# Start the App
root.mainloop()