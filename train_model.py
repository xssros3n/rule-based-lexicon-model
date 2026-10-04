import json
import fasttext
import random
import os

print("Generating dataset...")
with open("profanity_lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

safe_words = ["anmol", "maa", "bhai", "dost", "yaar", "hello", "hi", "kaisa", "hai", "kaha", "ho", "tu", "tum", "aap", "khana", "khaya", "kya", "kar", "raha"]
safe_sentences = [
    "kaisa hai bhai", "maa kaisi hai", "anmol kaha hai", "hello dost", "tu kya kar raha hai",
    "khana khaya?", "bhai kahan ho", "yaar aaj milte hain", "kya baat hai", "tu pagal hai kya",
    "bhai anmol kal aayega", "maa ne khana banaya", "aap kaise ho", "hi hello", "dost se milna hai"
]

train_data = []

# Generate safe data
for _ in range(5000):
    # Random safe sentence
    if random.random() > 0.5:
        sentence = random.choice(safe_sentences)
    else:
        # Construct random safe sentence
        sentence = " ".join(random.sample(safe_words, k=random.randint(2, 5)))
    train_data.append(f"__label__safe {sentence}")

# Generate abusive data
for word in lexicon.keys():
    # Exact word
    train_data.append(f"__label__abusive {word}")
    
    # In sentences
    for _ in range(20):
        # Insert the abusive word randomly in a safe sentence
        safe_base = random.choice(safe_sentences).split()
        insert_idx = random.randint(0, len(safe_base))
        safe_base.insert(insert_idx, word)
        train_data.append(f"__label__abusive {' '.join(safe_base)}")

random.shuffle(train_data)

with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_data))

print(f"Dataset generated with {len(train_data)} samples.")
print("Training FastText model...")

model = fasttext.train_supervised(
    input="train.txt", 
    epoch=25, 
    lr=0.1, 
    wordNgrams=2,
    dim=50
)

# Quantize the model to reduce size (for Render)
model.quantize(input="train.txt", retrain=True, qnorm=True)

model_path = "moderation_model.ftz"
model.save_model(model_path)
print(f"Model saved to {model_path} (Size: {os.path.getsize(model_path) / 1024:.2f} KB)")
