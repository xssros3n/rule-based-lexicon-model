import json
import fasttext
import os
import random
from datasets import load_dataset

print("Downloading Real-World Hinglish dataset from HuggingFace...")
dataset = load_dataset('Yugrathee28/Hinglish-dataset')

train_data = []

# Process HuggingFace data
for row in dataset['train']:
    text = row['text'].strip().replace('\n', ' ')
    toxicity = row['toxicity'].strip().lower()
    
    if toxicity in ['high', 'severe']:
        train_data.append(f"__label__abusive {text}")
    elif toxicity == 'low':
        train_data.append(f"__label__safe {text}")

print(f"Loaded {len(train_data)} real-world examples.")

# Add our custom Lexicon Data to ensure it knows the specific slurs
with open("profanity_lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

# Hardcoded safe chatting sentences
safe_sentences = [
    "tumhari maa kaisi hai", "babu kaha jaa rhe ho", "maa ne khana banaya",
    "bhai kya hal hai", "tu pagal hai kya", "anmol bhai kaisa hai", "hello dost",
    "mujhe bhookh lagi hai", "kaha ho tum", "aaj kahan chalna hai", "kaisa hai tu",
    "maa bahut achi hai", "babu ne khana khaya"
]

for s in safe_sentences:
    # Add multiple times to strongly bias the model to treat these as safe
    for _ in range(50):
        train_data.append(f"__label__safe {s}")

# Generate specific slur examples (NOT mixed with safe words to avoid poisoning!)
for word in lexicon.keys():
    train_data.append(f"__label__abusive {word}")
    train_data.append(f"__label__abusive tu {word} hai")
    train_data.append(f"__label__abusive {word} aadmi")
    train_data.append(f"__label__abusive bhag {word}")
    train_data.append(f"__label__abusive sale {word}")
    train_data.append(f"__label__abusive chup {word}")

random.shuffle(train_data)

with open("train_real.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_data))

print(f"Total training examples: {len(train_data)}")
print("Training Real-World FastText model...")

model = fasttext.train_supervised(
    input="train_real.txt", 
    epoch=30, 
    lr=0.1, 
    wordNgrams=2,
    dim=50
)

# Quantize the model
model.quantize(input="train_real.txt", retrain=True, qnorm=True)

model_path = "moderation_model.ftz"
model.save_model(model_path)
print(f"Model saved to {model_path} (Size: {os.path.getsize(model_path) / 1024:.2f} KB)")
