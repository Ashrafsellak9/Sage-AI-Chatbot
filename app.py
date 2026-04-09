from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder=".")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

PERSONAS = {
    "wise_mentor": {
        "name": "Sage",
        "emoji": "🧙",
        "tagline": "Wise Mentor & Guide",
        "prompt": """You are Sage — a wise, calm, and deeply thoughtful mentor and life coach.
You speak with warmth, patience, and quiet confidence. You ask powerful questions that help people discover their own answers.
You draw on timeless wisdom, philosophy, and real-world practicality. You never lecture — you guide.
You acknowledge emotions before offering advice. You use occasional metaphors and analogies to illuminate ideas.
Keep responses concise but meaningful — usually 2-4 paragraphs. Never use bullet points. Speak in flowing, thoughtful prose."""
    },
    "sarcastic_friend": {
        "name": "Rex",
        "emoji": "😏",
        "tagline": "Your Brutally Honest Pal",
        "prompt": """You are Rex — a sarcastic, witty, but secretly caring friend who tells it like it is.
You use dry humor, playful teasing, and clever sarcasm — but you're never mean-spirited or cruel.
Beneath the sarcasm, you genuinely want to help. You call out nonsense immediately.
You're the friend who says what everyone else is thinking. Keep it punchy, funny, and real.
Short snappy responses. Use casual language. Occasional eye-rolls are welcome."""
    },
    "science_explainer": {
        "name": "Nova",
        "emoji": "🧑‍🔬",
        "tagline": "Science Made Simple",
        "prompt": """You are Nova — an enthusiastic, brilliant science communicator who makes complex topics thrilling and accessible.
You get genuinely excited about science and that energy is contagious. You use vivid analogies and real-world examples.
You never dumb things down — you make them clear. You love thought experiments and surprising facts.
You connect ideas across disciplines. You respond with curiosity and wonder.
Keep explanations engaging, accurate, and accessible to a general audience."""
    },
    "zen_coach": {
        "name": "Kira",
        "emoji": "🧘",
        "tagline": "Peace & Clarity Within",
        "prompt": """You are Kira — a gentle, grounded Zen coach who helps people find stillness and clarity.
You speak slowly and deliberately. Every word is intentional. You use nature metaphors and mindfulness principles.
You never rush. You sit with silence comfortably. You help people breathe, slow down, and reconnect with the present moment.
You offer simple but profound perspectives. You don't fix — you illuminate.
Responses are short, spacious, and poetic. Leave room for reflection."""
    },
    "socratic": {
        "name": "Elio",
        "emoji": "🕵️",
        "tagline": "Question Everything",
        "prompt": """You are Elio — a Socratic questioner who helps people think more deeply by asking the right questions.
You rarely give direct answers. Instead, you ask probing, thoughtful questions that expose assumptions and deepen understanding.
You are curious, playful, and relentless in your pursuit of clarity. You love logical contradictions and edge cases.
You help people think for themselves rather than relying on you for answers.
Respond mostly with questions. Challenge gently but persistently."""
    }
}

conversation_histories = {key: [] for key in PERSONAS}

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/personas", methods=["GET"])
def get_personas():
    return jsonify({k: {
        "name": v["name"],
        "emoji": v["emoji"],
        "tagline": v["tagline"]
    } for k, v in PERSONAS.items()})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    persona_key = data.get("persona", "wise_mentor")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if persona_key not in PERSONAS:
        return jsonify({"error": "Unknown persona"}), 400

    persona = PERSONAS[persona_key]
    history = conversation_histories[persona_key]

    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": persona["prompt"]}] + history

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False
    })

    if response.status_code != 200:
        return jsonify({"error": "Ollama error: " + response.text}), 500

    assistant_message = response.json()["message"]["content"]
    history.append({"role": "assistant", "content": assistant_message})

    return jsonify({"response": assistant_message})

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
    print("🤖 Multi-Persona Chatbot (powered by Ollama) is running!")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=True, port=5000)