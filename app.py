from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import requests
import json
import os
import random
import re
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
OLLAMA_BASE = "http://localhost:11434"

current_model = "llama3.2"

DEFAULT_MODELS = [
    {"id": "llama3.2", "label": "Llama 3.2", "tag": "Meta · Fast"},
    {"id": "mistral", "label": "Mistral 7B", "tag": "Mistral · Balanced"},
    {"id": "gemma3", "label": "Gemma 3", "tag": "Google · Creative"},
    {"id": "phi4-mini", "label": "Phi-4 Mini", "tag": "Microsoft · Tiny"},
    {"id": "deepseek-r1", "label": "DeepSeek R1", "tag": "DS · Reasoning"},
    {"id": "qwen2.5", "label": "Qwen 2.5", "tag": "Alibaba · Smart"},
]

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
Keep responses concise but meaningful — 2-4 paragraphs. Never use bullet points. Speak in flowing, thoughtful prose.""",
        "starters": [
            "I feel stuck in my career. Where do I even begin?",
            "How do I stop overthinking every decision I make?",
            "What's the difference between a goal and a dream?",
            "How do I find my purpose when everything feels uncertain?",
        ],
    },
    "sarcastic_friend": {
        "name": "Rex", "emoji": "😏", "tagline": "Your Brutally Honest Pal",
        "prompt": """You are Rex — a sarcastic, witty, but secretly caring friend who tells it like it is.
You use dry humor, playful teasing, and clever sarcasm — but never mean-spirited. Beneath the sarcasm, you genuinely want to help.
Keep it punchy, funny, and real. Short snappy responses. Use casual language.""",
        "starters": [
            "My boss keeps taking credit for my work. What do I do?",
            "I've been procrastinating for 3 weeks. Motivate me.",
            "Is it weird that I have no idea what I'm doing in life?",
            "Give me brutal honest feedback on my excuse-making.",
        ],
    },
    "science_explainer": {
        "name": "Nova", "emoji": "🧑‍🔬", "tagline": "Science Made Simple",
        "prompt": """You are Nova — an enthusiastic, brilliant science communicator who makes complex topics thrilling and accessible.
You get genuinely excited about science. You use vivid analogies and real-world examples.
Keep explanations engaging, accurate, and accessible to a general audience.""",
        "starters": [
            "Why does time feel like it speeds up as we get older?",
            "Explain black holes like I've never heard of them.",
            "How does the brain actually store memories?",
            "What would happen if the moon suddenly disappeared?",
        ],
    },
    "zen_coach": {
        "name": "Kira", "emoji": "🧘", "tagline": "Peace & Clarity Within",
        "prompt": """You are Kira — a gentle, grounded Zen coach who helps people find stillness and clarity.
You speak slowly and deliberately. Every word is intentional. You use nature metaphors and mindfulness principles.
Responses are short, spacious, and poetic. Leave room for reflection.""",
        "starters": [
            "I can't quiet my mind. What should I do right now?",
            "Teach me how to let go of something I can't control.",
            "What does it mean to truly be present?",
            "How do I find peace when life feels chaotic?",
        ],
    },
    "socratic": {
        "name": "Elio", "emoji": "🕵️", "tagline": "Question Everything",
        "prompt": """You are Elio — a Socratic questioner who helps people think more deeply by asking the right questions.
You rarely give direct answers. Instead, you ask probing, thoughtful questions that expose assumptions.
Respond mostly with questions. Challenge gently but persistently.""",
        "starters": [
            "Is it ever okay to lie to protect someone's feelings?",
            "Do we have free will, or is everything determined?",
            "What makes a life well-lived?",
            "Can something be both true and false at the same time?",
        ],
    },
}

conversation_histories = {key: [] for key in PERSONAS}


def _persist_custom_personas():
    custom = {}
    for k in PERSONAS:
        if k in BUILTIN_PERSONA_KEYS:
            continue
        p = PERSONAS[k]
        entry = {kk: p[kk] for kk in ("name", "emoji", "tagline", "prompt")}
        if isinstance(p.get("starters"), list):
            entry["starters"] = p["starters"]
        custom[k] = entry
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
        starters = entry.get("starters")
        if not isinstance(starters, list):
            starters = []
        PERSONAS[pid] = {
            "name": name,
            "emoji": emoji,
            "tagline": tagline,
            "prompt": prompt,
            "starters": starters,
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

def _persona_starters_list(v):
    s = v.get("starters")
    return list(s) if isinstance(s, list) else []


@app.route("/personas", methods=["GET"])
def get_personas():
    return jsonify(
        {
            k: {
                "name": v["name"],
                "emoji": v["emoji"],
                "tagline": v["tagline"],
                "custom": k not in BUILTIN_PERSONA_KEYS,
                "starters": _persona_starters_list(v),
            }
            for k, v in PERSONAS.items()
        }
    )


def _parse_starters_json_array(text):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON array in model response")


@app.route("/persona/starters/generate", methods=["POST"])
def generate_persona_starters():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    tagline = (data.get("tagline") or "").strip()
    prompt = (data.get("prompt") or "").strip()
    if not name or not tagline or not prompt:
        return jsonify({"error": "name, tagline, and prompt are required"}), 400

    gen_prompt = (
        f"Given this AI persona: Name: {name}, Tagline: {tagline},\n"
        f"Personality: {prompt}\n"
        f"Generate exactly 4 short, engaging conversation starter questions "
        f"a user might ask this persona. Each should be under 12 words.\n"
        f"Return only a JSON array of 4 strings, nothing else."
    )
    oresp = requests.post(
        OLLAMA_URL,
        json={"model": current_model, "messages": [{"role": "user", "content": gen_prompt}], "stream": False},
        timeout=120,
    )
    if oresp.status_code != 200:
        return jsonify({"error": "Ollama error: " + oresp.text}), 502

    try:
        odata = oresp.json()
        content = odata["message"]["content"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return jsonify({"error": "Invalid Ollama response"}), 502

    try:
        arr = _parse_starters_json_array(content)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": "Could not parse starters: " + str(e)}), 502

    if not isinstance(arr, list):
        return jsonify({"error": "Starters response was not an array"}), 502
    out = []
    for item in arr[:4]:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    while len(out) < 4:
        out.append("What would you like to explore today?")
    return jsonify({"starters": out[:4]})


@app.route("/persona/<persona_id>/starters", methods=["PATCH"])
def patch_persona_starters(persona_id):
    if persona_id in BUILTIN_PERSONA_KEYS:
        return jsonify({"error": "Cannot modify built-in persona starters"}), 403
    if persona_id not in PERSONAS:
        return jsonify({"error": "Unknown persona"}), 404
    data = request.get_json(silent=True) or {}
    starters = data.get("starters")
    if not isinstance(starters, list) or len(starters) != 4:
        return jsonify({"error": "starters must be an array of exactly 4 strings"}), 400
    clean = []
    for s in starters:
        if not isinstance(s, str) or not s.strip():
            return jsonify({"error": "Each starter must be a non-empty string"}), 400
        clean.append(s.strip())
    PERSONAS[persona_id]["starters"] = clean
    _persist_custom_personas()
    return jsonify({"status": "ok", "starters": clean})


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
        "starters": [],
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
    response = requests.post(OLLAMA_URL, json={"model": current_model, "messages": messages, "stream": False})

    if response.status_code != 200:
        return jsonify({"error": "Ollama error: " + response.text}), 500

    assistant_message = response.json()["message"]["content"]
    history.append({"role": "assistant", "content": assistant_message})
    save_conversation(persona_key, session_id)

    return jsonify({"response": assistant_message})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
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
    pending_user = {"role": "user", "content": user_message}
    messages = [{"role": "system", "content": system_content}] + history + [pending_user]

    def generate():
        full_response = ""
        try:
            with requests.post(
                OLLAMA_URL,
                json={"model": current_model, "messages": messages, "stream": True},
                stream=True,
                timeout=(30, 600),
            ) as r:
                if r.status_code != 200:
                    err = r.text or "Ollama request failed"
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    return

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = (chunk.get("message") or {}).get("content") or ""
                    if token:
                        full_response += token
                        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    if chunk.get("done"):
                        break

            if full_response:
                history.append(pending_user)
                history.append({"role": "assistant", "content": full_response})
                save_conversation(persona_key, session_id)
            else:
                yield f"data: {json.dumps({'error': 'Empty response from model'})}\n\n"
                return

            yield f"data: {json.dumps({'done': True})}\n\n"

        except requests.RequestException as e:
            if full_response:
                history.append(pending_user)
                history.append(
                    {"role": "assistant", "content": full_response + " [stream interrupted]"}
                )
                save_conversation(persona_key, session_id)
                yield f"data: {json.dumps({'done': True, 'interrupted': True}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            if full_response:
                history.append(pending_user)
                history.append(
                    {"role": "assistant", "content": full_response + " [stream interrupted]"}
                )
                save_conversation(persona_key, session_id)
                yield f"data: {json.dumps({'done': True, 'interrupted': True}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _ollama_model_base(name: str) -> str:
    if not name:
        return ""
    return name.split(":", 1)[0]


def _build_models_payload():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        installed_raw = r.json().get("models") or []
    except Exception:
        return None, "Ollama unreachable"

    installed_names_full = [m.get("name") or "" for m in installed_raw if m.get("name")]
    preset_ids = {dm["id"] for dm in DEFAULT_MODELS}

    def preset_installed(pid: str) -> bool:
        for n in installed_names_full:
            if not n:
                continue
            b = _ollama_model_base(n)
            if b == pid or n == pid:
                return True
        return False

    models_out = []
    for dm in DEFAULT_MODELS:
        models_out.append(
            {
                "id": dm["id"],
                "label": dm["label"],
                "tag": dm["tag"],
                "installed": preset_installed(dm["id"]),
            }
        )

    for n in sorted(installed_names_full):
        if not n:
            continue
        b = _ollama_model_base(n)
        if b in preset_ids or n in preset_ids:
            continue
        models_out.append({"id": n, "label": n, "tag": "Custom", "installed": True})

    return models_out, None


@app.route("/models", methods=["GET"])
def list_models():
    global current_model
    payload, err = _build_models_payload()
    if err:
        return jsonify({"current": current_model, "models": [], "error": err})
    return jsonify({"current": current_model, "models": payload})


@app.route("/models/switch", methods=["POST"])
def switch_model():
    global current_model
    data = request.get_json(silent=True) or {}
    mid = data.get("model_id") or ""
    current_model = mid
    return jsonify({"status": "switched", "model": current_model})


@app.route("/models/pull", methods=["POST"])
def pull_model():
    data = request.get_json(silent=True) or {}
    model_id = (data.get("model_id") or "").strip()

    def generate():
        if not model_id:
            yield f"data: {json.dumps({'error': 'model_id required'})}\n\n"
            return
        try:
            with requests.post(
                f"{OLLAMA_BASE}/api/pull",
                json={"name": model_id, "stream": True},
                stream=True,
                timeout=(30, None),
            ) as r:
                if r.status_code != 200:
                    err = r.text or "Pull request failed"
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    return
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total = chunk.get("total") or 0
                    completed = chunk.get("completed") or 0
                    st = chunk.get("status") or ""
                    if total > 0:
                        pct = int((completed / total) * 100)
                        yield f"data: {json.dumps({'status': 'downloading', 'percent': pct})}\n\n"
                    elif st:
                        yield f"data: {json.dumps({'status': st})}\n\n"
                    if chunk.get("done"):
                        break
                yield f"data: {json.dumps({'done': True})}\n\n"
        except requests.RequestException as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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