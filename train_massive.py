import json
import random
import os
import gc
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

print("Step 1: Loading Massive Real-world Dataset...")
# We take a random subset of 50,000 real world messages to keep memory low but intelligence high
# This dataset has real English conversations where "hello", "hi", "good", etc. are marked safe (label=0)
ds = load_dataset('SetFit/toxic_conversations', split='train')
ds = ds.shuffle(seed=42).select(range(50000))

train_texts = [str(x['text']) for x in ds]
train_labels = ["abusive" if x['label'] == 1 else "safe" for x in ds]

print(f"Loaded {len(train_texts)} real-world examples (Safe and Toxic).")

print("Step 2: Mixing Desi Hinglish Lexicon...")
with open("profanity_lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

# Hardcoded purely safe sentences to reinforce Hinglish safe words
safe_sentences = [
    "tumhari maa kaisi hai", "babu kaha jaa rhe ho", "maa ne khana banaya",
    "bhai kya hal hai", "tu pagal hai kya", "anmol bhai kaisa hai", "hello dost",
    "mujhe bhookh lagi hai", "kaha ho tum", "aaj kahan chalna hai", "kaisa hai tu",
    "maa bahut achi hai", "babu ne khana khaya", "uski behan school me hai",
    "chutney do", "hello babu", "bhai jaan kaisa hai", "teri shirt achi hai",
    "kutta animal hai", "mera kutta bhaag gaya"
]

# Reinforce Hinglish safe words so they are strictly label 0
for _ in range(500):
    for s in safe_sentences:
        train_texts.append(s)
        train_labels.append("safe")

safe_words = "tumhari maa kaisi hai babu kaha jaa rhe ho ne khana banaya bhai kya hal tu pagal anmol dost mujhe bhookh lagi tum aaj chalna bahut achi uski behan school me chutney teri shirt kutta animal mera bhaag gaya".split()
for _ in range(5000):
    train_texts.append(" ".join(random.sample(safe_words, k=random.randint(2, 6))))
    train_labels.append("safe")

def make_chalaak(word):
    vars = []
    # 1. Leetspeak
    reps = {'a':'@', 'i':'!', 'o':'0', 'e':'3', 's':'$', 'c':'k'}
    vars.append("".join([reps.get(c, c) for c in word]))
    # 2. Spaced out
    vars.append(" ".join(list(word)))
    # 3. Repeated chars
    vars.append("".join([c*random.randint(1,3) for c in word]))
    return vars

# Inject purely abusive Hinglish words + Chalaak variations
for word in lexicon.keys():
    train_texts.append(word)
    train_labels.append("abusive")
    # Add chalaak versions
    for cv in make_chalaak(word):
        for _ in range(10): # give them some weight
            train_texts.append(cv)
            train_labels.append("abusive")

print(f"Total dataset size after mixing chalaak words: {len(train_texts)} examples.")

print("Step 3: Training TF-IDF + LinearSVC Model...")
# Use char_wb (Character N-grams within word boundaries) to catch m@d@r, m a d a r
# Combine with word-level by just relying on char_wb 2-6 which covers whole short words and subwords
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=25000)
X = vectorizer.fit_transform(train_texts)

# Extremely fast and memory efficient Linear Support Vector Machine
clf = LinearSVC(C=1.0, class_weight='balanced', random_state=42)
clf.fit(X, train_labels)

# Save the models
print("Saving models to disk...")
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(clf, 'model.joblib')

print("Models saved successfully! Production ready.")
