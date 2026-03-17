"""
api.py — Servidor Flask que expone el pipeline de recomendación al frontend.

Coloca este archivo en la misma carpeta que main.py y ejecuta:
    pip install flask flask-cors
    python api.py
"""

import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from feedback import initialize_cpt_counts, apply_feedback, load_cpt_counts, save_cpt_counts
from graph_builder import load_model
from LLM_agent import (
    classify_intent,
    converse,
    extract_attributes_llm,
    get_time_daytype,
    infer_with_bn,
)
from content_fetcher import TMDBContentFetcher
from smart_alternative import should_skip_to_next_genre, get_next_different_genre
from main import fetch_real_content, try_next_alternative

# ════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ════════════════════════════════════════════════════════════════

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# Frontend en http://localhost:5000
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

# Modelo Bayesiano
MODEL_PATH  = Path(__file__).parent / "output/model.pkl"
COUNTS_PATH = Path(__file__).parent / "output/cpt_counts.json"

model      = load_model(str(MODEL_PATH))
cpt_counts = initialize_cpt_counts(model, virtual_sample_size=100)

if COUNTS_PATH.exists():
    try:
        cpt_counts = load_cpt_counts(COUNTS_PATH)
        print("✅ Datos de aprendizaje cargados")
    except Exception as e:
        print(f"⚠️  No se pudieron cargar los counts: {e}")

# TMDB
try:
    content_fetcher = TMDBContentFetcher()
    print("✅ TMDB conectado")
except Exception as e:
    content_fetcher = None
    print(f"⚠️  TMDB no disponible ({e}) — modo solo-género activo")

# ── Estado de sesión en memoria ──────────────────────────────────
# Cada sesión de navegador comparte este estado.
session_state = {
    "atributes_bn":      {},
    "candidates":        {},
    "last_recommendation": None,
    "user_feedback":     None,
    "real_content":      [],
    "content_index":     0,
    "content_available": False,
}
conversation_history = []


# ════════════════════════════════════════════════════════════════
# ENDPOINT PRINCIPAL
# POST /api/chat
# Body:    { "message": "texto del usuario" }
# Returns: { "message", "action", "item", "content" }
# ════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(silent=True) or {}
    mensaje = data.get("message", "").strip()

    if not mensaje:
        return jsonify({"error": "Campo 'message' vacío"}), 400

    # ── 1. Clasificar intención ──────────────────────────────────
    intent = classify_intent(mensaje)
    print(f"[intent] {intent}")

    # ── 2. Lógica según intención ────────────
    if intent == "RECOMMEND":
        atributes = extract_attributes_llm(mensaje)
        time_of_day, day_type = get_time_daytype()
        atributes["TimeOfDay"] = time_of_day
        atributes["DayType"]   = day_type

        session_state["atributes_bn"] = atributes
        bn_result = infer_with_bn(session_state, model)
        session_state["candidates"] = bn_result

        if content_fetcher:
            real_content = fetch_real_content(bn_result, content_fetcher, limit=10)
            session_state["real_content"]      = real_content
            session_state["content_index"]     = 0
            session_state["content_available"] = len(real_content) > 0

            if real_content:
                session_state["last_recommendation"] = {
                    "ProgramType":  bn_result["ProgramType"],
                    "ProgramGenre": bn_result["ProgramGenre"],
                    "content":      real_content[0],
                }
            else:
                session_state["last_recommendation"] = {
                    "ProgramType":  bn_result["ProgramType"],
                    "ProgramGenre": bn_result["ProgramGenre"],
                }
        else:
            session_state["real_content"]      = []
            session_state["content_available"] = False
            session_state["last_recommendation"] = {
                "ProgramType":  bn_result["ProgramType"],
                "ProgramGenre": bn_result["ProgramGenre"],
            }

        session_state["user_feedback"] = None

    elif intent == "ALTERNATIVE":
        session_state["user_feedback"] = "rejected"
        apply_feedback(model, cpt_counts, session_state, learning_rate=50)

        skip_genre, rejected_genre = should_skip_to_next_genre(mensaje, session_state)

        if skip_genre and rejected_genre:
            if content_fetcher:
                has_alt = get_next_different_genre(session_state, content_fetcher, rejected_genre)
                session_state["content_available"] = has_alt
            else:
                bn_result     = session_state.get("candidates", {})
                genre_ranking = bn_result.get("genre_ranking", [])
                for genre in genre_ranking:
                    if genre != rejected_genre:
                        bn_result["ProgramGenre"] = genre
                        session_state["candidates"] = bn_result
                        if session_state.get("last_recommendation"):
                            session_state["last_recommendation"]["ProgramGenre"] = genre
                        break
        else:
            if content_fetcher and session_state.get("content_available"):
                has_alt = try_next_alternative(session_state, content_fetcher)
                session_state["content_available"] = has_alt
            else:
                bn_result     = session_state.get("candidates", {})
                genre_ranking = bn_result.get("genre_ranking", [])
                current_genre = bn_result.get("ProgramGenre")
                try:
                    idx = genre_ranking.index(current_genre)
                    if idx + 1 < len(genre_ranking):
                        next_genre = genre_ranking[idx + 1]
                        bn_result["ProgramGenre"] = next_genre
                        session_state["candidates"] = bn_result
                        if session_state.get("last_recommendation"):
                            session_state["last_recommendation"]["ProgramGenre"] = next_genre
                except (ValueError, IndexError):
                    pass

    elif intent == "FEEDBACK_POS":
        session_state["user_feedback"] = "accepted"
        apply_feedback(model, cpt_counts, session_state, learning_rate=50)

    elif intent == "FEEDBACK_NEG":
        session_state["user_feedback"] = "rejected"
        skip_genre, rejected_genre = should_skip_to_next_genre(mensaje, session_state)
        if skip_genre and rejected_genre:
            apply_feedback(model, cpt_counts, session_state, learning_rate=50)
            if content_fetcher:
                has_alt = get_next_different_genre(session_state, content_fetcher, rejected_genre)
                session_state["content_available"] = has_alt
        else:
            apply_feedback(model, cpt_counts, session_state, learning_rate=50)

    # SMALLTALK / OTHER → no hace nada con el estado

    # ── 3. Generar respuesta conversacional ──────────────────────
    raw_response = converse(mensaje, session_state, conversation_history)

    try:
        response = json.loads(raw_response)
    except json.JSONDecodeError:
        print(f"[JSON error] {raw_response}")
        return jsonify({"error": "Error interno al parsear respuesta"}), 500

    # ── 4. Actualizar historial ──────────────────────────────────
    conversation_history.append({"role": "user",      "content": mensaje})
    conversation_history.append({"role": "assistant", "content": response.get("message", "")})

    # Guardar counts periódicamente
    _save_counts()

    # ── 5. Construir respuesta para el frontend ──────────────────
    last_rec = session_state.get("last_recommendation") or {}
    content  = last_rec.get("content")  # objeto TMDB completo o None

    # Si el LLM devolvió un content_id, buscar el contenido correspondiente
    content_id   = response.get("content_id")
    real_content = session_state.get("real_content", [])
    if content_id and real_content:
        matching = next((c for c in real_content if c.get("id") == content_id), None)
        if matching:
            content = matching
            if session_state.get("last_recommendation") is not None:
                session_state["last_recommendation"]["content"] = matching

    return jsonify({
        "message": response.get("message", ""),
        "action":  response.get("action",  "SMALLTALK"),
        "item":    response.get("item"),
        "content": content,   # el frontend usa este objeto para la tarjeta
    })


# ── Endpoint para resetear la sesión────────
@app.route("/api/reset", methods=["POST"])
def reset():
    session_state.update({
        "atributes_bn":        {},
        "candidates":          {},
        "last_recommendation": None,
        "user_feedback":       None,
        "real_content":        [],
        "content_index":       0,
        "content_available":   False,
    })
    conversation_history.clear()
    return jsonify({"ok": True})


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _save_counts():
    """Guarda los CPT counts en disco (igual que main.py al salir)."""
    try:
        save_cpt_counts(cpt_counts, COUNTS_PATH)
    except Exception as e:
        print(f"⚠️  No se pudieron guardar los counts: {e}")


# ════════════════════════════════════════════════════════════════
# ARRANQUE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎬  Servidor de recomendación iniciado")
    print("   → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)