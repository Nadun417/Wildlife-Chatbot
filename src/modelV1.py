import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK data (only needed the first time)
nltk.download("punkt")

# ─── STAGE 1: Load intents ────────────────────────────────────────────────────

stemmer = LancasterStemmer()

with open("data/intents.json") as f:
    data = json.load(f)

words = []       # every word found across all patterns
classes = []     # every unique tag
documents = []   # pairs of (pattern word list, tag)
ignore = ["?", "!", ".", ","]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# ─── STAGE 2: Stem and clean words ───────────────────────────────────────────

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(set(words))    # remove duplicates and sort alphabetically
classes = sorted(set(classes))

print(f"Vocabulary size : {len(words)} unique stemmed words")
print(f"Intent classes  : {len(classes)} unique tags")
print(f"Training samples: {len(documents)} patterns")

# ─── STAGE 3: Create bag of words training data ───────────────────────────────

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [stemmer.stem(w.lower()) for w in doc[0]]

    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = list(training[:, 0])    # input:  bag of words arrays
train_y = list(training[:, 1])    # output: correct tag as one-hot array

# ─── STAGE 4: Build neural network ───────────────────────────────────────────

model = Sequential()

# Input layer → first hidden layer (128 neurons)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))

# Second hidden layer (64 neurons)
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

# Output layer — one neuron per intent tag
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.summary()

# ─── STAGE 5: Train the model ─────────────────────────────────────────────────

print("\nTraining started...")

history = model.fit(
    np.array(train_x),
    np.array(train_y),
    epochs=300,
    batch_size=5,
    verbose=1
)

final_accuracy = history.history["accuracy"][-1] * 100
print(f"\nTraining complete. Final accuracy: {final_accuracy:.2f}%")

# ─── STAGE 6: Save model and vocabulary ──────────────────────────────────────

model.save("models/chatbot_model_v1.h5")

with open("models/words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("models/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("Model saved to   : models/chatbot_model_v1.h5")
print("Vocabulary saved : models/words.pkl")
print("Classes saved    : models/classes.pkl")