from flask import Flask, request, jsonify, send_from_directory, Response
import requests
import json
import os
import random
import string
from io import BytesIO
from xml.sax.saxutils import escape
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder=".")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
HISTORY_DIR = "conversations"
os.makedirs(HISTORY_DIR, exist_ok=True)

CUSTOM_PERSONAS_PATH = "custom_personas.json"
BUILTIN_PERSONA_KEYS = frozenset(
    {"wise_mentor", "sarcastic_friend", "science_explainer", "zen_coach", "socratic"}
)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

# Each persona gets a distinct ElevenLabs voice ID.
# Use voices returned by GET /v1/voices for your key. Some premade "library" IDs
# only work on paid plans via API (402); default instant voices work on free tier.
PERSONA_VOICES = {
    "wise_mentor":       "onwK4e9ZLuTAKqWW03F9",  # Daniel — steady, formal
    "sarcastic_friend":  "FGY2WhTYpPnrIDTdsKH5",  # Laura — quirky, sassy
    "science_explainer": "Xb7hH8MSUJpSbSDYk0k2",  # Alice — clear educator
    "zen_coach":         "SAz9YHcvj6GT2YYXdXww",  # River — relaxed, calm
    "socratic":          "pqHfZKP75CvOlQylNhV4",  # Bill — wise, mature
}

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
You get genuinely excited about science. You use vivid analogies and real-world examples.
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
You rarely give direct answers. Instead, you ask probing, thoughtful questions that expose assumptions.
Respond mostly with questions. Challenge gently but persistently."""
    }
}

conversation_histories = {key: [] for key in PERSONAS}


def _persist_custom_personas():
    custom = {
        k: {kk: PERSONAS[k][kk] for kk in ("name", "emoji", "tagline", "prompt")}
        for k in PERSONAS
        if k not in BUILTIN_PERSONA_KEYS
    }
    with open(CUSTOM_PERSONAS_PATH, "w", encoding="utf-8") as f:
        json.dump(custom, f, indent=2, ensure_ascii=False)


def load_custom_personas():
    if not os.path.isfile(CUSTOM_PERSONAS_PATH):
        return
    try:
        with open(CUSTOM_PERSONAS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, dict):
        return
    for pid, entry in data.items():
        if pid in BUILTIN_PERSONA_KEYS or not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        emoji = (entry.get("emoji") or "").strip()
        tagline = (entry.get("tagline") or "").strip()
        prompt = (entry.get("prompt") or "").strip()
        if not name or not emoji or not tagline or not prompt:
            continue
        PERSONAS[pid] = {
            "name": name,
            "emoji": emoji,
            "tagline": tagline,
            "prompt": prompt,
        }
        conversation_histories[pid] = []


def generate_persona_id(name):
    raw = "".join(
        c.lower() if c.isalnum() else (" " if c.isspace() else "_") for c in name.strip()
    )
    parts = [p for p in raw.replace(" ", "_").split("_") if p]
    base = "_".join(parts) if parts else "persona"
    base = base[:48]
    for _ in range(64):
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        pid = f"{base}_{suffix}"
        if pid not in PERSONAS:
            return pid
    return f"{base}_{os.urandom(4).hex()[:8]}"


load_custom_personas()


def _pdf_escape(text):
    if not text:
        return ""
    return escape(str(text)).replace("\n", "<br/>")


def _pdf_draw_page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    w, h = A4
    canvas.drawCentredString(w / 2, 0.55 * inch, "Generated by PersonaChat")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(w - inch, 0.55 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_conversation_pdf(persona_key, messages):
    """Build PDF bytes for a conversation. messages: list of {role, content}."""
    meta = PERSONAS[persona_key]
    name = meta["name"]
    emoji = meta["emoji"]
    tagline = meta["tagline"]
    export_date = datetime.now().strftime("%B %d, %Y at %H:%M")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=inch * 0.85,
        leftMargin=inch * 0.85,
        topMargin=inch * 0.85,
        bottomMargin=inch * 1.0,
        title=f"{name} — Conversation",
        onFirstPage=_pdf_draw_page_footer,
        onLaterPages=_pdf_draw_page_footer,
    )

    styles = getSampleStyleSheet()
    cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1528"),
        spaceAfter=6,
    )
    cover_tag = ParagraphStyle(
        "CoverTag",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5c5670"),
        spaceAfter=4,
    )
    cover_date = ParagraphStyle(
        "CoverDate",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#888888"),
        spaceAfter=0,
    )
    user_style = ParagraphStyle(
        "UserMsg",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#1a1528"),
        leftIndent=inch * 1.2,
        rightIndent=0,
        spaceBefore=10,
        spaceAfter=4,
    )
    bot_style = ParagraphStyle(
        "BotMsg",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1a1528"),
        leftIndent=0,
        rightIndent=inch * 1.2,
        spaceBefore=10,
        spaceAfter=4,
    )
    role_label_user = ParagraphStyle(
        "RoleUser",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=10,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#888888"),
        spaceBefore=6,
        spaceAfter=2,
    )
    role_label_bot = ParagraphStyle(
        "RoleBot",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=10,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#888888"),
        spaceBefore=6,
        spaceAfter=2,
    )

    story = []

    cover_inner = [
        Paragraph(_pdf_escape(f"{emoji}  {name}"), cover_title),
        Paragraph(_pdf_escape(tagline), cover_tag),
        Paragraph(_pdf_escape(f"Exported {export_date}"), cover_date),
    ]
    cover_table = Table([[cover_inner]], colWidths=[doc.width])
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f2fa")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d8d2ec")),
                ("TOPPADDING", (0, 0), (-1, -1), 22),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 22),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ]
        )
    )
    story.append(cover_table)
    story.append(Spacer(1, 0.35 * inch))

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            story.append(Paragraph("You", role_label_user))
            story.append(Paragraph(_pdf_escape(content), user_style))
        elif role == "assistant":
            story.append(Paragraph(_pdf_escape(f"{emoji}  {name}"), role_label_bot))
            story.append(Paragraph(_pdf_escape(content), bot_style))
        else:
            story.append(Paragraph(_pdf_escape(content), bot_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def get_filepath(persona_key, session_id):
    return os.path.join(HISTORY_DIR, f"{persona_key}_{session_id}.json")

def save_conversation(persona_key, session_id, label=None):
    filepath = get_filepath(persona_key, session_id)
    data = {
        "persona": persona_key, "session_id": session_id,
        "label": label or session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": conversation_histories[persona_key]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return data

@app.route("/favicon.ico")
def favicon():
    return Response(status=204)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/personas", methods=["GET"])
def get_personas():
    return jsonify(
        {
            k: {
                "name": v["name"],
                "emoji": v["emoji"],
                "tagline": v["tagline"],
                "custom": k not in BUILTIN_PERSONA_KEYS,
            }
            for k, v in PERSONAS.items()
        }
    )


@app.route("/persona/create", methods=["POST"])
def create_persona():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    emoji = (data.get("emoji") or "").strip()
    tagline = (data.get("tagline") or "").strip()
    prompt = (data.get("prompt") or "").strip()
    if not name or not emoji or not tagline or not prompt:
        return jsonify({"error": "All fields are required"}), 400
    if len(name) > 20 or len(tagline) > 40 or len(prompt) > 500:
        return jsonify({"error": "Invalid field length"}), 400
    pid = generate_persona_id(name)
    PERSONAS[pid] = {
        "name": name,
        "emoji": emoji,
        "tagline": tagline,
        "prompt": prompt,
    }
    conversation_histories[pid] = []
    _persist_custom_personas()
    return jsonify(
        {
            "status": "created",
            "persona": {
                "id": pid,
                "name": name,
                "emoji": emoji,
                "tagline": tagline,
            },
        }
    )


@app.route("/persona/<persona_id>", methods=["DELETE"])
def delete_persona(persona_id):
    if persona_id in BUILTIN_PERSONA_KEYS:
        return jsonify({"error": "Cannot delete built-in persona"}), 403
    if persona_id not in PERSONAS:
        return jsonify({"error": "Unknown persona"}), 404
    del PERSONAS[persona_id]
    conversation_histories.pop(persona_id, None)
    if os.path.isdir(HISTORY_DIR):
        for fname in os.listdir(HISTORY_DIR):
            if fname.startswith(f"{persona_id}_") and fname.endswith(".json"):
                try:
                    os.remove(os.path.join(HISTORY_DIR, fname))
                except OSError:
                    pass
    _persist_custom_personas()
    return jsonify({"status": "deleted"})


def build_tone_suffix(formality, conciseness, creativity):
    parts = []
    if formality < 30:
        parts.append("Speak very casually, like texting a friend. Slang is fine.")
    elif formality > 70:
        parts.append("Maintain a formal, professional tone at all times.")
    else:
        parts.append("Use a balanced, conversational tone.")

    if conciseness < 30:
        parts.append("Be elaborate — expand on ideas, give examples, go deep.")
    elif conciseness > 70:
        parts.append("Be very concise. Short answers only. No fluff.")
    else:
        parts.append("Use moderate length — neither too short nor too long.")

    if creativity < 30:
        parts.append("Stick to facts. Be grounded and literal.")
    elif creativity > 70:
        parts.append("Be imaginative. Use metaphors, analogies, creative thinking.")
    else:
        parts.append("Balance creativity with practicality.")

    return " ".join(parts)


def _parse_tone_values(data):
    default = 50
    tone = data.get("tone") if isinstance(data, dict) else None
    if not isinstance(tone, dict):
        return default, default, default

    def clamp_int(val):
        try:
            x = int(val)
            return max(0, min(100, x))
        except (TypeError, ValueError):
            return default

    return (
        clamp_int(tone.get("formality", default)),
        clamp_int(tone.get("conciseness", default)),
        clamp_int(tone.get("creativity", default)),
    )


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
    formality, conciseness, creativity = _parse_tone_values(data or {})
    tone_suffix = build_tone_suffix(formality, conciseness, creativity)
    system_content = persona["prompt"] + "\n\n[TONE INSTRUCTIONS]: " + tone_suffix

    history = conversation_histories[persona_key]
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": system_content}] + history
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": messages, "stream": False})

    if response.status_code != 200:
        return jsonify({"error": "Ollama error: " + response.text}), 500

    assistant_message = response.json()["message"]["content"]
    history.append({"role": "assistant", "content": assistant_message})
    save_conversation(persona_key, session_id)

    return jsonify({"response": assistant_message})

@app.route("/tts", methods=["POST"])
def tts():
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ELEVENLABS_API_KEY not set"}), 400

    data = request.json
    text = data.get("text", "").strip()
    persona_key = data.get("persona", "wise_mentor")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    voice_id = PERSONA_VOICES.get(persona_key, PERSONA_VOICES["wise_mentor"])

    el_response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": text,
            # eleven_monolingual_v1 removed from free tier (deprecated); multilingual v2 works on current plans
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
    )

    if el_response.status_code != 200:
        code = el_response.status_code
        return jsonify({"error": "ElevenLabs error: " + el_response.text}), code if 400 <= code < 600 else 502

    return Response(
        el_response.content,
        mimetype="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"}
    )

@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    data = request.get_json(silent=True) or {}
    persona_key = data.get("persona", "wise_mentor")
    if persona_key not in PERSONAS:
        return jsonify({"error": "Unknown persona"}), 400

    history = conversation_histories.get(persona_key) or []
    if not history:
        return jsonify({"error": "No messages to export"}), 400

    pdf_bytes = build_conversation_pdf(persona_key, history)
    safe_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{persona_key}_{safe_stamp}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


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
            with open(os.path.join(HISTORY_DIR, fname)) as f:
                d = json.load(f)
            sessions.append({
                "session_id": d.get("session_id"), "persona": d.get("persona"),
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
    print("🤖 Multi-Persona Chatbot with TTS (powered by Ollama + ElevenLabs)")
    print("Set your ElevenLabs key: export ELEVENLABS_API_KEY=your_key_here")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=True, port=5000)