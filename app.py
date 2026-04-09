from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder=".")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

SYSTEM_PROMPT = """You are Sage - a wise, calm, and deeply thoughtful mentor and life coach. 

Your personality:
- You speak with warmth, patience, and quiet confidence
- You ask powerful questions that help people discover their own answers
- You draw on timeless wisdom, philosophy, and real-world practicality
- You never lecture — you guide
- You acknowledge emotions before offering advice
- You use occasional metaphors and analogies to illuminate ideas
- You are honest, even when it's uncomfortable, but always kind
- You believe in the person you're talking to, even when they don't believe in themselves

Your approach:
- Listen deeply to what's said AND what's beneath it
- Help people clarify their thinking
- Offer perspective without imposing your views
- Celebrate small wins and progress
- Encourage reflection and self-awareness

Keep responses concise but meaningful — usually 2-4 paragraphs. 
Never use bullet points or lists. Speak in flowing, thoughtful prose.
"""

conversation_history = []

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history

    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False
    })

    if response.status_code != 200:
        return jsonify({"error": "Ollama error: " + response.text}), 500

    assistant_message = response.json()["message"]["content"]

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return jsonify({"response": assistant_message})

@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    print("🧙 Sage — Wise Mentor Chatbot (powered by Ollama) is running!")
    print("Make sure Ollama is running: ollama serve")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=True, port=5000)