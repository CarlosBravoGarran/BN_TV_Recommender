import json
import os
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from bn_recommender import recommend_gender
from graph_builder import load_model
from feedback import initialize_cpt_counts, apply_feedback

# Logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# ANSI colors for quick visual cues per intervention type
COLOR_RESET = "\033[0m"
ACTION_COLORS = {
    "RECOMMEND": "\033[92m",   # green
    "ASK": "\033[96m",         # cyan
    "ALTERNATIVE": "\033[93m", # yellow
    "SMALLTALK": "\033[95m",   # magenta
    "FEEDBACK": "\033[94m",    # blue
}
INTENT_COLORS = {
    "RECOMMEND": "\033[92m",
    "ALTERNATIVE": "\033[93m",
    "FEEDBACK_POS": "\033[94m",
    "FEEDBACK_NEG": "\033[91m",
    "SMALLTALK": "\033[95m",
    "OTHER": "\033[90m",
}
BN_LOG_COLOR = "\033[90m"  # dim gray for internal BN logs


def colorize(text: str, color_code: str) -> str:
    if not color_code:
        return text
    return f"{color_code}{text}{COLOR_RESET}"


# Load .env
env_path = ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not defined in .env file")

client = OpenAI(api_key=api_key)

# SYSTEM PROMPTS

SYSTEM_PROMPT = """
Eres un asistente de televisión diseñado para ayudar a personas mayores a decidir qué ver.
Tu comportamiento depende del state que recibirás en cada turno.

El state contiene:
- atributes_bn (atributos extraídos para el BN)
- candidatos del recomendador
- la última recomendación del turno anterior
- feedback del usuario ("accepted" o "rejected")

Tu tarea es:
1. Interpretar la intención del usuario.
2. Decidir si debes:
   - recomendar algo
   - explicar la recomendación
   - ofrecer una alternativa
   - pedir más información
   - dar feedback de la recomendación
3. Generar un mensaje conversacional claro y amable.
4. No inventar programas que no existan.
5. SIEMPRE usar los candidatos del state si existen.
6. Si no hay candidatos, recomendar por género ("comedia", "documental", etc).
7. Si el usuario ha rechazado algo, ofrecer una alternativa distinta.
8. Recomienda solamente un género o título a la vez.
9. Responder SIEMPRE con un JSON con los campos:

{
 "action": "RECOMMEND" | "ASK" | "ALTERNATIVE" | "SMALLTALK" | "FEEDBACK",
 "message": "mensaje conversacional para el usuario",
 "item": "título recomendado o null"
}

NUNCA respondas fuera del JSON.
"""

# Intent classifier prompt

INTENT_PROMPT = """
Eres un clasificador de intención. Devuelve SOLO un JSON válido:

{
 "intent": "RECOMMEND" | "ALTERNATIVE" | "FEEDBACK_POS" | "FEEDBACK_NEG" | "SMALLTALK" | "OTHER"
}

Reglas:
- RECOMMEND: El usuario pide recomendación, sugerencia, qué ver, etc.
- ALTERNATIVE: El usuario pide otra opción o rechaza la recomendación previa ("otra", "no esa", etc).
- FEEDBACK_POS: Acepta la recomendación ("me gusta", "perfecto", "vale").
- FEEDBACK_NEG: La rechaza explícitamente ("no quiero", "eso no").
- SMALLTALK: Conversación trivial ("hola", "gracias", etc.)
- OTHER: Todo lo que no encaje.

No añadas texto fuera del JSON.
"""

# Attribute extraction prompt

EXTRACTION_PROMPT = """
Eres un modelo extractor. Devuelve SOLO un JSON válido con estos campos,
alineados exactamente con los nodos de una Red Bayesiana:

{
 "UserAge": ...,
 "UserGender": ...,
 "HouseholdType": ...,
 "TimeOfDay": ...,
 "DayType": ...,
 "ProgramType": ...,
 "ProgramGenre": ...,
 "ProgramDuration": ...
}

Reglas:
- Si no se menciona un atributo, usa null.
- No añadas texto fuera del JSON.
- No uses markdown.
- Usa EXACTAMENTE los nombres de los campos indicados.

Convenciones:
- UserAge: "young" (18-35), "adult" (36-55), "senior" (56+)
- UserGender: "male" | "female"
- HouseholdType: "single" | "couple" | "family"

- TimeOfDay:
  - "morning" (07:00-12:00)
  - "afternoon" (12:00-20:00)
  - "night" (20:00-07:00)

- DayType: "weekday" | "weekend"

- ProgramType: "movie" | "series" | "news" | "documentary" | "entertainment"
- ProgramGenre: "comedy" | "drama" | "horror" | "romance" | "news" | "documentary" | "entertainment"
- ProgramDuration:
  - "short" (<30 min)
  - "medium" (30-60 min)
  - "long" (>60 min)

Si el usuario menciona un tipo de programa pero no la duración, infiere la duración típica.
Si no hay información suficiente, usa null.
"""


# Intent classifier function

def classify_intent(user_message: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )
    data = json.loads(resp.choices[0].message.content)
    return data["intent"]

# Attribute extraction

def extract_attributes_llm(user_message: str) -> dict:

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content.strip())

# BN inference

def recommend_by_genre(state: dict) -> dict:

    attrs = state.get("atributes_bn") or {}

    filtered = {k: v for k, v in attrs.items() if v not in (None, "", "null")}
    print(colorize(f"Non-null attributes for BN: {filtered}", BN_LOG_COLOR))

    return filtered

def infer_with_bn(state: dict, model) -> list:

    attrs = recommend_by_genre(state)
    if not attrs:
        print(colorize("Not enough attributes for BN inference.", BN_LOG_COLOR))
        return []

    recommendations = recommend_gender(attrs, model)
    sorted_items = [g for g, _ in recommendations]

    print(colorize(f"BN Recommendations: {sorted_items}", BN_LOG_COLOR))
    return sorted_items

# Conversational LLM

def converse(user_message, state, history=None):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "STATE:\n" + json.dumps(state, indent=2)}
    ]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content

def get_time_daytype():
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()

    if 7 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 20:
        time_of_day = "afternoon"
    else:
        time_of_day = "night"

    if weekday < 5:
        day_type = "weekday"
    else:
        day_type = "weekend"

    return time_of_day, day_type

# MAIN LOOP

if __name__ == "__main__":
    
    history = []
    states_log = []

    state = {
        "atributes_bn": {},
        "candidates": [],
        "last_recommendation": None,
        "user_feedback": None
    }

    print("TV Assistant. Type 'exit' to quit.\n")

    model = load_model("main/outputs/model.pkl")
    cpt_counts = initialize_cpt_counts(model)

    while True:
        mensaje = input("User: ")

        if mensaje.lower().strip() == "exit":
            break

        # 1) Classify intent
        intent = classify_intent(mensaje)
        intent_msg = f"Detected intent: {intent}"
        print(colorize(intent_msg, INTENT_COLORS.get(intent, "")))

        # 2) Logic based on intent

        if intent == "RECOMMEND":
            atributes = extract_attributes_llm(mensaje)

            time_of_day, day_type = get_time_daytype()
            
            atributes["TimeOfDay"] = time_of_day
            atributes["DayType"] = day_type

            state["atributes_bn"] = atributes
            state["candidates"] = infer_with_bn(state, model)

        elif intent == "ALTERNATIVE":
            state["user_feedback"] = "rejected"
            apply_feedback(model, cpt_counts, state)

        elif intent == "FEEDBACK_POS":
            state["user_feedback"] = "accepted"
            apply_feedback(model, cpt_counts, state)
        elif intent == "FEEDBACK_NEG":
            state["user_feedback"] = "rejected"
            apply_feedback(model, cpt_counts, state)

        elif intent == "SMALLTALK":
            pass  # do not modify BN

        elif intent == "OTHER":
            pass  # do not modify BN

        # Save state
        states_log.append(json.loads(json.dumps(state)))

        # 3) Final conversation

        raw_response = converse(mensaje, state, history)

        try:
            response = json.loads(raw_response)
        except:
            print("JSON ERROR:", raw_response)
            continue

        action = response.get("action")
        message = response.get("message")
        item = response.get("item")

        action_color = ACTION_COLORS.get(action, "")
        action_tag = action if action else "UNKNOWN"
        assistant_line = f"Assistant ({action_tag}): {message}"
        print(colorize(assistant_line, action_color))

        history.append({"role": "user", "content": mensaje})
        history.append({"role": "assistant", "content": message})

        if item:
            state["last_recommendation"] = item

        if action == "ALTERNATIVE":
            state["user_feedback"] = "rejected"
        elif action == "RECOMMEND":
            state["user_feedback"] = None

    # Final state save

    save_path = Path(__file__).parent / "states.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(states_log, f, indent=2, ensure_ascii=False)

    print(f"All STATES saved to states.json")
