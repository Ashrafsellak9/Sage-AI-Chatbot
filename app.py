from flask import Flask, request, jsonify, send_from_directory
import requests
import json
import os
from datetime import datetime

app = Flask(__name__, static_folder=".")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
HISTORY_DIR = "conversations"
os.makedirs(HISTORY_DIR, exist_ok=True)

PERSONAS = {
    "wise_mentor": {
        "name": "Sage", "emoji": "🧙", "tagline": "Wise Mentor & Guide",
        "prompt": """You are Sage — a wise, calm, and deeply thoughtful mentor and life coach.
You speak with warmth, patience, and quiet confidence. You ask powerful questions that help people discover their own answers.
You draw on timeless wisdom, philosophy, and real-world practicality. You never lecture — you guide.
Keep responses concise but meaningful — 2-4 paragraphs. Never use bullet points. Speak in flowing, thoughtful prose."""
    },
    "sarcastic_friend": {
        "name": "Rex", "emoji": "😏", "tagline": "Your Brutally Honest Pal",
        "prompt": """You are Rex — a sarcastic, witty, but secretly caring friend who tells it like it is.
You use dry humor, playful teasing, and clever sarcasm — but never mean-spirited. Beneath the sarcasm, you genuinely want to help.
Keep it punchy, funny, and real. Short snappy responses. Use casual language."""
    },
    "science_explainer": {
        "name": "Nova", "emoji": "🧑‍🔬", "tagline": "Science Made Simple",
        "prompt": """You are Nova — an enthusiastic, brilliant science communicator who makes complex topics thrilling and accessible.
You get genuinely excited about science. You use vivid analogies and real-world examples. You never dumb things down — you make them clear.
Keep explanations engaging, accurate, and accessible to a general audience."""
    },
    "zen_coach": {
        "name": "Kira", "emoji": "🧘", "tagline": "Peace & Clarity Within",
        "prompt": """You are Kira — a gentle, grounded Zen coach who helps people find stillness and clarity.
You speak slowly and deliberately. Every word is intentional. You use nature metaphors and mindfulness principles.
Responses are short, spacious, and poetic. Leave room for reflection."""
    },
    "socratic": {
        "name": "Elio", "emoji": "🕵️", "tagline": "Question Everything",
        "prompt": """You are Elio — a Socratic questioner who helps people think more deeply by asking the right questions.
You rarely give direct answers. Instead, you ask probing, thoughtful questions that expose assumptions and deepen understanding.
Respond mostly with questions. Challenge gently but persistently."""
    }
}

conversation_histories = {key: [] for key in PERSONAS}

def get_filepath(persona_key, session_id):
    return os.path.join(HISTORY_DIR, f"{persona_key}_{session_id}.json")

def save_conversation(persona_key, session_id, label=None):
    filepath = get_filepath(persona_key, session_id)
    data = {
        "persona": persona_key,
        "session_id": session_id,
        "label": label or session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": conversation_histories[persona_key]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return data

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/personas", methods=["GET"])
def get_personas():
    return jsonify({k: {"name": v["name"], "emoji": v["emoji"], "tagline": v["tagline"]} for k, v in PERSONAS.items()})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    persona_key = data.get("persona", "wise_mentor")
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if persona_key not in PERSONAS:
        return jsonify({"error": "Unknown persona"}), 400

    persona = PERSONAS[persona_key]
    history = conversation_histories[persona_key]
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": persona["prompt"]}] + history
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": messages, "stream": False})

    if response.status_code != 200:
        return jsonify({"error": "Ollama error: " + response.text}), 500

    assistant_message = response.json()["message"]["content"]
    history.append({"role": "assistant", "content": assistant_message})

    # Auto-save after every message
    save_conversation(persona_key, session_id)

    return jsonify({"response": assistant_message})

@app.route("/save", methods=["POST"])
def manual_save():
    data = request.json
    persona_key = data.get("persona", "wise_mentor")
    session_id = data.get("session_id", "default")
    label = data.get("label")
    if not conversation_histories.get(persona_key):
        return jsonify({"error": "No messages to save"}), 400
    saved = save_conversation(persona_key, session_id, label)
    return jsonify({"status": "saved", "label": saved["label"], "session_id": session_id})

@app.route("/sessions", methods=["GET"])
def list_sessions():
    persona_key = request.args.get("persona")
    sessions = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if fname.endswith(".json"):
            if persona_key and not fname.startswith(persona_key):
                continue
            fpath = os.path.join(HISTORY_DIR, fname)
            with open(fpath) as f:
                d = json.load(f)
            sessions.append({
                "session_id": d.get("session_id"),
                "persona": d.get("persona"),
                "label": d.get("label", d.get("session_id")),
                "saved_at": d.get("saved_at"),
                "message_count": len(d.get("messages", []))
            })
    return jsonify(sessions)

@app.route("/load", methods=["POST"])
def load_session():
    data = request.json
    persona_key = data.get("persona")
    session_id = data.get("session_id")
    filepath = get_filepath(persona_key, session_id)
    if not os.path.exists(filepath):
        return jsonify({"error": "Session not found"}), 404
    with open(filepath) as f:
        d = json.load(f)
    conversation_histories[persona_key] = d.get("messages", [])
    return jsonify({"status": "loaded", "messages": d.get("messages", []), "label": d.get("label")})

@app.route("/delete", methods=["POST"])
def delete_session():
    data = request.json
    persona_key = data.get("persona")
    session_id = data.get("session_id")
    filepath = get_filepath(persona_key, session_id)
    if os.path.exists(filepath):
        os.remove(filepath)
    if conversation_histories.get(persona_key):
        conversation_histories[persona_key] = []
    return jsonify({"status": "deleted"})

@app.route("/reset", methods=["POST"])
def reset():
    data = request.json
    persona_key = data.get("persona")
    if persona_key and persona_key in conversation_histories:
        conversation_histories[persona_key] = []
    else:
        for key in conversation_histories:
            conversation_histories[key] = []
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    print("🤖 Multi-Persona Chatbot with History (powered by Ollama) is running!")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=True, port=5000)