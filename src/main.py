import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import load_model

# ─── STAGE 1: Load saved model and data ──────────────────────────────────────

stemmer = LancasterStemmer()

# Load intents (needed to look up responses)
with open("data/intents.json") as f:
    data = json.load(f)

# Load the trained neural network
# Change to chatbot_model_v2.h5 when V2 is ready
model = load_model("models/chatbot_model_v1.h5")

# Load vocabulary and classes saved during training
with open("models/words.pkl", "rb") as f:
    words = pickle.load(f)

with open("models/classes.pkl", "rb") as f:
    classes = pickle.load(f)

# ─── STAGE 2: Process user input into bag of words ───────────────────────────

def process_input(user_input):
    # Tokenize the input into individual words
    input_words = nltk.word_tokenize(user_input)

    # Stem each word to its root form — same as done during training
    input_words = [stemmer.stem(w.lower()) for w in input_words]

    # Build the bag of words array
    # 1 if the word from vocabulary appears in input, 0 if not
    bag = [0] * len(words)
    for w in input_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# ─── STAGE 3: Predict the intent ─────────────────────────────────────────────

def predict_intent(user_input):
    # Convert input to bag of words
    bag = process_input(user_input)

    # Get probability predictions from the neural network
    # verbose=0 suppresses the progress bar output
    predictions = model.predict(np.array([bag]), verbose=0)[0]

    # Only keep predictions above the error threshold
    # This prevents the bot from confidently answering unrelated questions
    ERROR_THRESHOLD = 0.25
    results = [
        [i, p] for i, p in enumerate(predictions) if p > ERROR_THRESHOLD
    ]

    # Sort by probability — highest first
    results.sort(key=lambda x: x[1], reverse=True)

    # If nothing passed the threshold the bot didn't understand
    if not results:
        return None

    # Return the tag name of the highest probability intent
    return classes[results[0][0]]

# ─── STAGE 4: Pick a response ─────────────────────────────────────────────────

def get_response(intent_tag):
    # If intent is None (below threshold) return fallback message
    if intent_tag is None:
        return "I'm not sure I understand. Could you rephrase that?"

    # Find the matching intent in intents.json by tag
    for intent in data["intents"]:
        if intent["tag"] == intent_tag:
            # Randomly pick one of the responses for variety
            return random.choice(intent["responses"])

    # Fallback if tag somehow not found
    return "I'm not sure I understand. Could you rephrase that?"

# ─── STAGE 5: Chat loop ───────────────────────────────────────────────────────

def chat():
    print("=" * 50)
    print("  Sri Lanka Wildlife Guide Chatbot")
    print("  Type 'quit' to exit")
    print("=" * 50)
    print()

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Skip empty input
        if not user_input:
            continue

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print("Bot: Goodbye! Have a great wildlife adventure!")
            break

        # Predict intent and get response
        intent = predict_intent(user_input)
        response = get_response(intent)
        print(f"Bot: {response}\n")

# Run the chatbot only when this file is executed directly
if __name__ == "__main__":
    chat()