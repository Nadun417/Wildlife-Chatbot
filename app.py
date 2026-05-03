from flask import Flask, render_template, request, jsonify
import json
import pickle
import random
import os
import subprocess
from threading import Lock
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

RETRAIN_THRESHOLD = 5    # retrain after every 5 corrections
INTENTS_PATH = "data/intents.json"
FEEDBACK_LOG_PATH = "data/feedback_log.json"
MODEL_PATH = "models/chatbot_model_v1.h5"

# ─── Globals ──────────────────────────────────────────────────────────────────

stemmer = LancasterStemmer()
training_lock = Lock()
is_training = False
correction_counter = 0

model = None
words = None
classes = None
data = None

# ─── Loading functions ────────────────────────────────────────────────────────

def load_chatbot():
    """Load (or reload) the model, vocabulary, classes and intents file."""
    global model, words, classes, data

    with open(INTENTS_PATH) as f:
        data = json.load(f)
    model = load_model(MODEL_PATH)
    with open("models/words.pkl", "rb") as f:
        words = pickle.load(f)
    with open("models/classes.pkl", "rb") as f:
        classes = pickle.load(f)
    print("Chatbot loaded successfully.")

# Load everything when Flask starts
load_chatbot()

# Initialise feedback log if it doesn't exist
if not os.path.exists(FEEDBACK_LOG_PATH):
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump({"corrections": [], "reinforcements": []}, f, indent=4)

# ─── Prediction helpers ───────────────────────────────────────────────────────

def process_input(user_input):
    input_words = nltk.word_tokenize(user_input)
    input_words = [stemmer.stem(w.lower()) for w in input_words]
    bag = [0] * len(words)
    for w in input_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_intent(user_input):
    bag = process_input(user_input)
    predictions = model.predict(np.array([bag]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, p] for i, p in enumerate(predictions) if p > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return None
    return classes[results[0][0]]

def get_response(intent_tag):
    if intent_tag is None:
        return "I'm not sure I understand. Could you rephrase that?"
    for intent in data["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understand. Could you rephrase that?"

# ─── Feedback / learning helpers ──────────────────────────────────────────────

def add_pattern_to_intent(tag, new_pattern, new_response=None):
    """Append a new pattern (and optional response) to an existing tag,
    OR create a new tag if it doesn't exist."""
    with open(INTENTS_PATH) as f:
        intents_data = json.load(f)

    found = False
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            if new_pattern not in intent["patterns"]:
                intent["patterns"].append(new_pattern)
            if new_response and new_response not in intent["responses"]:
                intent["responses"].append(new_response)
            found = True
            break

    if not found:
        # Create a brand new intent
        new_intent = {
            "tag": tag,
            "patterns": [new_pattern],
            "responses": [new_response] if new_response else ["I'll get back to you on that soon!"]
        }
        intents_data["intents"].append(new_intent)

    with open(INTENTS_PATH, "w") as f:
        json.dump(intents_data, f, indent=4)

def log_feedback(feedback_type, entry):
    """Save feedback details to the log file for record keeping."""
    with open(FEEDBACK_LOG_PATH) as f:
        log = json.load(f)
    log[feedback_type].append(entry)
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(log, f, indent=4)

def retrain_model():
    """Run modelV1.py and reload the chatbot."""
    global is_training
    with training_lock:
        is_training = True
        print("Retraining started...")
        subprocess.run(["python", "src/modelV1.py"], check=False)
        load_chatbot()
        is_training = False
        print("Retraining complete.")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message.strip():
        return jsonify({"response": "Please type something.", "tag": None})
    intent = predict_intent(user_message)
    response = get_response(intent)
    return jsonify({
        "response": response,
        "tag": intent,
        "user_message": user_message
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    """Handle thumbs-up reinforcement feedback."""
    payload = request.json
    user_message = payload.get("user_message", "")
    predicted_tag = payload.get("tag", None)

    if predicted_tag and user_message:
        add_pattern_to_intent(predicted_tag, user_message)
        log_feedback("reinforcements", {
            "user_message": user_message,
            "tag": predicted_tag
        })

    return jsonify({"status": "ok", "message": "Thanks for the feedback!"})

@app.route("/correct", methods=["POST"])
def correct():
    """Handle thumbs-down correction feedback."""
    global correction_counter
    payload = request.json
    user_message = payload.get("user_message", "")
    correct_tag = payload.get("correct_tag", "").strip()
    new_response = payload.get("new_response", "").strip() or None

    if not user_message or not correct_tag:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    add_pattern_to_intent(correct_tag, user_message, new_response)
    log_feedback("corrections", {
        "user_message": user_message,
        "correct_tag": correct_tag,
        "new_response": new_response
    })

    correction_counter += 1
    retrained = False

    if correction_counter >= RETRAIN_THRESHOLD:
        correction_counter = 0
        retrain_model()
        retrained = True

    return jsonify({
        "status": "ok",
        "retrained": retrained,
        "corrections_until_retrain": RETRAIN_THRESHOLD - correction_counter
    })

@app.route("/tags", methods=["GET"])
def get_tags():
    """Return all current intent tags so frontend can populate dropdown."""
    return jsonify({"tags": sorted(classes)})

@app.route("/training_status", methods=["GET"])
def training_status():
    return jsonify({"is_training": is_training})

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)