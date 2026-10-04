import json
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

print("Generating Pure Word-level Dataset...")

with open("profanity_lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

# Hardcoded purely safe sentences that were causing issues
safe_sentences = [
    "tumhari maa kaisi hai", "babu kaha jaa rhe ho", "maa ne khana banaya",
    "bhai kya hal hai", "tu pagal hai kya", "anmol bhai kaisa hai", "hello dost",
    "mujhe bhookh lagi hai", "kaha ho tum", "aaj kahan chalna hai", "kaisa hai tu",
    "maa bahut achi hai", "babu ne khana khaya", "uski behan school me hai",
    "chutney do", "hello babu", "bhai jaan kaisa hai", "teri shirt achi hai",
    "kutta animal hai", "mera kutta bhaag gaya"
]

train_texts = []
train_labels = []

# 1. Generate Safe Data (Label 0)
for _ in range(500):
    for s in safe_sentences:
        train_texts.append(s)
        train_labels.append("safe")

# We can also add random permutations of safe words to be very robust
safe_words = "tumhari maa kaisi hai babu kaha jaa rhe ho ne khana banaya bhai kya hal tu pagal anmol dost mujhe bhookh lagi tum aaj chalna bahut achi uski behan school me chutney teri shirt kutta animal mera bhaag gaya".split()
for _ in range(5000):
    train_texts.append(" ".join(random.sample(safe_words, k=random.randint(2, 6))))
    train_labels.append("safe")

# 2. Generate Abusive Data (Label 1)
for word in lexicon.keys():
    # Only the slur
    train_texts.append(word)
    train_labels.append("abusive")
    # Slur in sentences
    for _ in range(20):
        # We wrap the slur with safe words. 
        # Crucially, TF-IDF will learn the slur ITSELF is toxic, not the surrounding words.
        safe_base = random.choice(safe_sentences).split()
        insert_idx = random.randint(0, len(safe_base))
        safe_base.insert(insert_idx, word)
        train_texts.append(" ".join(safe_base))
        train_labels.append("abusive")

print(f"Generated {len(train_texts)} training examples.")
print("Training TF-IDF + LinearSVC Model...")

# We use character n-grams ONLY within words, or purely word n-grams
# Word n-grams up to 2 (bigrams) are excellent for contextual slurs without cross-word character confusion
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
X = vectorizer.fit_transform(train_texts)

# Linear Support Vector Classification (SVM) - extremely fast and robust for text
clf = LinearSVC(C=1.0, class_weight='balanced', random_state=42)
clf.fit(X, train_labels)

# Save the models
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(clf, 'model.joblib')

print("Models saved successfully: vectorizer.joblib, model.joblib")
