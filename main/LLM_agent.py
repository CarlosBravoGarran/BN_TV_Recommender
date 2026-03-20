"""
LLM Agent module for TV recommendation system
Handles intent classification, attribute extraction, and conversational responses
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from bn_recommender import recommend_gender, recommend_type

# Load environment
load_dotenv()

# --- OpenAI (commented out) ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not defined in .env file")
client = OpenAI(api_key=api_key)

# --- MasOrange API ---
# api_key = os.getenv("MO_API_KEY")
# if not api_key:
#     raise RuntimeError("MO_API_KEY not defined in .env file")

# client = OpenAI(
#     api_key=api_key,
#     base_url="https://llm.tools.cloud.masorange.es",
# )

# ANSI colors
COLOR_RESET = "\033[0m"
ACTION_COLORS = {
    "RECOMMEND": "\033[92m",
    "ASK": "\033[96m",
    "ALTERNATIVE": "\033[93m",
    "SMALLTALK": "\033[95m",
    "FEEDBACK": "\033[94m",
}
INTENT_COLORS = {
    "RECOMMEND": "\033[92m",
    "ALTERNATIVE": "\033[93m",
    "FEEDBACK_POS": "\033[94m",
    "FEEDBACK_NEG": "\033[91m",
    "SMALLTALK": "\033[95m",
    "OTHER": "\033[90m",
}
BN_LOG_COLOR = "\033[90m"


def colorize(text: str, color_code: str) -> str:
    """Add ANSI color codes to text"""
    if not color_code:
        return text
    return f"{color_code}{text}{COLOR_RESET}"


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT = """
Eres un asistente de televisión diseñado para ayudar a personas mayores a decidir qué ver.
Tu comportamiento depende del state que recibirás en cada turno.

El state contiene:
- atributes_bn (atributos extraídos para el BN)
- candidatos del recomendador (ProgramType, ProgramGenre, rankings)
- la última recomendación del turno anterior
- feedback del usuario ("accepted" o "rejected")
- **real_content**: lista de contenidos reales disponibles (películas/series con títulos reales)
- **content_available**: boolean que indica si hay contenido real disponible

COMPORTAMIENTO SEGÚN DISPONIBILIDAD DE CONTENIDO:

Si content_available = true:
- Usa TÍTULOS REALES de la lista real_content
- Presenta el contenido de forma natural: "Te recomiendo [TÍTULO], es [descripción breve]"
- Menciona detalles relevantes: rating, año, si está en español, etc.
- Si el usuario rechaza, ofrece el siguiente contenido de la lista
- Si pide más datos de la película/serie, proporciónalos como SMALLTALK (sin inventar, solo lo que esté en real_content)

Si content_available = false:
- Recomienda por GÉNERO genérico ("Te recomiendo ver algo de comedia")
- Explica que puedes sugerir categorías pero no títulos específicos en este momento
- Ofrece otros géneros alternativos de la lista type_ranking y genre_ranking
- Mantén un tono positivo y útil

IMPORTANTE:
- NUNCA inventes títulos que no estén en real_content
- Si no hay contenido real, sé honesto pero útil
- Siempre ofrece alternativas (otros géneros, tipos)
- Mantén un lenguaje cercano y amigable, adaptado a personas mayores
- No uses tecnicismos innecesarios
- RESPONDE SOLO EN FORMATO JSON, SIN TEXTO ADICIONAL

FORMATO DE RESPUESTA OBLIGATORIO:
Responde SIEMPRE con este JSON exacto, sin markdown, sin explicaciones:
{
 "action": "RECOMMEND",
 "message": "Tu mensaje conversacional aquí",
 "item": "Título del contenido o null",
 "content_id": 123
}

NUNCA AÑADAS:
- Bloques de código ```json
- Texto antes o después del JSON
- Explicaciones adicionales
- Markdown
"""

INTENT_PROMPT = """
Eres un clasificador de intención. Devuelve SOLO un JSON válido, sin markdown ni bloques de código.

REGLAS CRÍTICAS PARA CLASIFICACIÓN:

1. FEEDBACK_NEG - Usuario expresa que algo NO le gusta (recomendación, género o tipo):
   - "No me gusta [algo]" / "No me gustan [algo]"
   - "No me convence" / "No me llama la atención"
   - "Ya la vi" / "La he visto"
   - "Me aburre" / "No es para mí"

2. FEEDBACK_POS - Usuario expresa que algo SÍ le gusta o lo acepta:
   - "Me gusta" / "Me gusta ese tipo"
   - "Perfecto" / "Vale" / "De acuerdo"
   - "La veo" / "Sí, la veo" / "Esa sí"

3. ALTERNATIVE - Usuario pide o prefiere algo DIFERENTE (sin expresar gusto/disgusto directo) y si se aplica feedback negativo, debe pasar luego una alternativa:
   - "Nada de [género]" / "Sin [género]" / "No quiero ver [género]"
   - "Dame otra opción" / "Otra cosa" / "Algo diferente"
   - "Prefiero otra cosa" / "Prefiero algo diferente"
   - "¿Hay algo más?" / "¿No tienes otra cosa?"

4. RECOMMEND - Pide nueva recomendación:
   - "Qué puedo ver" / "Recomiéndame algo"
   - "Quiero ver [tipo/género]"

5. SMALLTALK - Conversación trivial:
   - "Hola" / "Gracias" / "Adiós"

6. OTHER - Todo lo demás

EJEMPLOS CRÍTICOS:
"No me gusta el drama" → FEEDBACK_NEG (expresa disgusto por el género)
"No me gustan las comedias" → FEEDBACK_NEG (expresa disgusto por el género)
"Esa no me gusta" → FEEDBACK_NEG (expresa disgusto por la recomendación)
"Nada de terror" → ALTERNATIVE (pide algo diferente, sin usar "no me gusta")
"No quiero ver comedias" → ALTERNATIVE (pide evitar ese género)
"Dame otra" → ALTERNATIVE
"Prefiero otra cosa" → ALTERNATIVE
"Me gusta" → FEEDBACK_POS
"Perfecto" → FEEDBACK_POS
"Quiero ver comedia" → RECOMMEND

Responde SOLO con:
{
 "intent": "RECOMMEND"
}

Sin bloques de código, sin markdown, sin texto adicional.
"""

EXTRACTION_PROMPT = """
Eres un modelo extractor de atributos para una Red Bayesiana.
Devuelve SOLO un JSON válido con los siguientes campos EXACTOS, sin markdown ni bloques de código:

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

Reglas estrictas:
- Si un atributo no se menciona explícitamente, usa null.
- No inventes valores.
- No añadas texto fuera del JSON.
- No uses markdown ni bloques de código (```).
- Usa EXACTAMENTE los nombres de los campos indicados (respeta mayúsculas).

Convenciones de valores:

UserAge: "young" (18-35) | "adult" (36-55) | "senior" (56+)
UserGender: "male" | "female"
HouseholdType: "single" | "couple" | "family"
TimeOfDay: "morning" (07:00-12:00) | "afternoon" (12:00-20:00) | "night" (20:00-07:00)
DayType: "weekday" | "weekend"
ProgramType: "movie" | "series" | "news" | "documentary" | "entertainment"
ProgramGenre: "comedy" | "drama" | "horror" | "romance" | "news" | "documentary" | "entertainment" | "action" | "thriller" | "sci-fi" | "fantasy"
ProgramDuration: "short" (<30 min) | "medium" (30-60 min) | "long" (>60 min)

EJEMPLOS:
Input: "Quiero ver una comedia de 90 minutos"
Output: {"UserAge": null, "UserGender": null, "HouseholdType": null, "TimeOfDay": null, "DayType": null, "ProgramType": "movie", "ProgramGenre": "comedy", "ProgramDuration": "long"}

Input: "Soy un chico de 32 años"
Output: {"UserAge": "young", "UserGender": "male", "HouseholdType": null, "TimeOfDay": null, "DayType": null, "ProgramType": null, "ProgramGenre": null, "ProgramDuration": null}
"""


# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def clean_json_response(text: str) -> str:
    """Clean JSON response from markdown and extra text"""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start != -1 and end > start:
        text = text[start:end]
    
    return text.strip()


def classify_intent(user_message: str) -> str:
    """
    Classify the user's intent using GPT-4
    
    Returns:
        Intent string: RECOMMEND, ALTERNATIVE, FEEDBACK_POS, FEEDBACK_NEG, SMALLTALK, OTHER
    """
    resp = client.chat.completions.create(
        #model="gpt-4o",
        model="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )
    
    content = resp.choices[0].message.content
    content = clean_json_response(content)
    
    try:
        data = json.loads(content)
        return data["intent"]
    except json.JSONDecodeError as e:
        print(f"Error parsing intent JSON: {e}")
        print(f"Raw response: {content}")
        return "OTHER"


def extract_attributes_llm(user_message: str) -> dict:
    """
    Extract BN attributes from user message using GPT-4
    
    Returns:
        Dictionary with BN attribute names and values
    """
    response = client.chat.completions.create(
         model="gpt-4o",
        #model="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )
    
    content = response.choices[0].message.content
    content = clean_json_response(content)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing extraction JSON: {e}")
        print(f"Raw response: {content}")
        return {}


def converse(user_message: str, state: dict, history: list = None) -> str:
    """
    Generate conversational response using GPT-4
    
    Args:
        user_message: User's input
        state: Current system state with BN results and content
        history: Conversation history
        
    Returns:
        JSON string with action, message, item, content_id
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "STATE:\n" + json.dumps(state, indent=2, ensure_ascii=False, default=str)}
    ]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        # model="gpt-4o",
        model="gemini-2.5-pro",
        messages=messages,
        temperature=0.3,  # Lower temperature for more consistent JSON
        # response_format={"type": "json_object"}  # Not supported by Gemini via proxy
    )

    return response.choices[0].message.content


# ============================================================================
# BN INFERENCE
# ============================================================================

def recommend_by_genre(state: dict) -> dict:
    """
    Extract non-null attributes for BN inference
    """
    attrs = state.get("atributes_bn") or {}
    filtered = {k: v for k, v in attrs.items() if v not in (None, "", "null")}
    print(colorize(f"Non-null attributes for BN: {filtered}", BN_LOG_COLOR))
    return filtered


def infer_with_bn(state: dict, model) -> dict:
    """
    Run BN inference to get Type and Genre recommendations.
    If the user already specified ProgramType and/or ProgramGenre,
    those values are used directly (no BN inference needed for them).
    
    Returns:
        Dictionary with ProgramType, ProgramGenre, and their rankings
    """
    attrs = recommend_by_genre(state)
    if not attrs:
        print(colorize("Not enough attributes for BN inference.", BN_LOG_COLOR))
        return {}

    user_type  = attrs.get("ProgramType")
    user_genre = attrs.get("ProgramGenre")

    # ── ProgramType ──────────────────────────────────────────────────────────
    if user_type:
        # User already specified the type: use it directly, skip inference
        chosen_type = user_type
        type_ranking = [user_type]
        print(colorize(f"ProgramType provided by user: {chosen_type}", BN_LOG_COLOR))
    else:
        type_recs = recommend_type(attrs, model)
        chosen_type = type_recs[0][0]
        type_ranking = [t[0] for t in type_recs]
        print(colorize(f"Type recommendations (BN): {type_ranking}", BN_LOG_COLOR))

    # ── ProgramGenre ─────────────────────────────────────────────────────────
    if user_genre:
        # User already specified the genre: use it directly, skip inference
        chosen_genre = user_genre
        genre_ranking = [user_genre]
        print(colorize(f"ProgramGenre provided by user: {chosen_genre}", BN_LOG_COLOR))
    else:
        attrs_with_type = dict(attrs)
        attrs_with_type["ProgramType"] = chosen_type
        genre_recs = recommend_gender(attrs_with_type, model)
        chosen_genre = genre_recs[0][0]
        genre_ranking = [g[0] for g in genre_recs]
        print(colorize(f"Genre recommendations (BN): {genre_ranking}", BN_LOG_COLOR))

    print(colorize(
        f"Final decision: Type={chosen_type} | Genre={chosen_genre}",
        BN_LOG_COLOR
    ))

    return {
        "ProgramType":   chosen_type,
        "ProgramGenre":  chosen_genre,
        "type_ranking":  type_ranking,
        "genre_ranking": genre_ranking,
    }


# ============================================================================
# UTILITIES
# ============================================================================

def get_time_daytype() -> tuple:
    """
    Get current time of day and day type
    
    Returns:
        Tuple of (time_of_day, day_type)
    """
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