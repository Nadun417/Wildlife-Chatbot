# Sri Lanka Wildlife Guide Chatbot

A neural network–powered chatbot that answers questions about Sri Lanka's national parks and wildlife. Users can ask about animals, entry fees, directions, best seasons, and safari activities across 7 major parks.

## Features

- Intent classification using a bag-of-words neural network (TensorFlow/Keras)
- Web interface built with Flask
- Thumbs-up / thumbs-down feedback system
- Auto-retraining triggered after every 5 corrections
- Command-line interface for quick testing

## Parks Covered

| Park | Highlight |
|---|---|
| Yala | Highest leopard density in the world |
| Udawalawe | Year-round large elephant herds |
| Wilpattu | Leopards near natural lakes (villus) |
| Minneriya | The Gathering — up to 300 elephants (Aug–Oct) |
| Horton Plains | World's End cliff, endemic highland birds |
| Bundala | Flamingos and migratory wetland birds |
| Sinharaja | UNESCO rainforest, nearly all endemic bird species |

## Project Structure

```
Wildlife Chatbot/
├── app.py               # Flask web application
├── src/
│   ├── main.py          # Command-line chatbot
│   └── modelV1.py       # Model training script
├── data/
│   └── intents.json     # Training data (patterns & responses)
├── models/              # Saved model files (generated after training)
│   ├── chatbot_model_v1.h5
│   ├── words.pkl
│   └── classes.pkl
├── templates/
│   └── index.html       # Web UI
├── static/
│   ├── css/style.css
│   └── js/chat.js
└── dependencies.bat     # Windows setup script
```

## Setup

### Requirements

- Python 3.11
- Windows (for `dependencies.bat`) — or install manually on other platforms

### 1. Install dependencies

Run the setup script (Windows):

```bat
dependencies.bat
```

This creates a virtual environment, installs all packages, and downloads NLTK data.

**Manual install (all platforms):**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install tensorflow nltk numpy flask
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 2. Train the model

```bash
python src/modelV1.py
```

This reads `data/intents.json`, trains a neural network for 300 epochs, and saves the model to `models/`.

### 3. Run the chatbot

**Web interface:**

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Command line:**

```bash
python src/main.py
```

## Model Architecture

| Layer | Details |
|---|---|
| Input | Bag-of-words vector (vocabulary size) |
| Dense + Dropout | 128 neurons, ReLU, 50% dropout |
| Dense + Dropout | 64 neurons, ReLU, 50% dropout |
| Output | Softmax over intent classes |

- Optimizer: SGD (lr=0.01, momentum=0.9, Nesterov)
- Loss: Categorical cross-entropy
- Epochs: 300

## Feedback & Retraining

- **Thumbs up** — reinforces the current pattern into `intents.json`
- **Thumbs down** — lets the user provide the correct intent tag and an optional response; after 5 corrections the model retrains automatically

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Send a message, get a response and intent tag |
| POST | `/feedback` | Submit a thumbs-up reinforcement |
| POST | `/correct` | Submit a correction with the right tag |
| GET | `/tags` | List all current intent tags |
| GET | `/training_status` | Check if retraining is in progress |
