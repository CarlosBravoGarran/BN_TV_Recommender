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

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not defined in .env file")

client = OpenAI(api_key=api_key)

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

Tu tarea es:
1. Interpretar la intención del usuario
2. Decidir si debes:
   - recomendar algo específico (de real_content)
   - recomendar por género (si no hay real_content)
   - explicar la recomendación
   - ofrecer una alternativa
   - pedir más información
   - reconocer feedback positivo/negativo
3. Generar un mensaje conversacional claro y amable
4. Responder SIEMPRE con un JSON con los campos:
{
 "action": "RECOMMEND" | "ASK" | "ALTERNATIVE" | "SMALLTALK" | "FEEDBACK",
 "message": "mensaje conversacional para el usuario",
 "item": "título exacto del contenido recomendado o null",
 "content_id": "ID del contenido de TMDB o null"
}

NUNCA respondas fuera del JSON.
"""

INTENT_PROMPT = """
Eres un clasificador de intención. Devuelve SOLO un JSON válido:

{
 "intent": "RECOMMEND" | "ALTERNATIVE" | "FEEDBACK_POS" | "FEEDBACK_NEG" | "SMALLTALK" | "OTHER"
}

Reglas:
- RECOMMEND: El usuario pide recomendación, sugerencia, qué ver, etc.
  Ejemplos: "qué puedo ver", "recomiéndame algo", "quiero ver una película"
  
- ALTERNATIVE: El usuario pide otra opción o rechaza la recomendación previa.
  Ejemplos: "otra cosa", "no esa", "dame más opciones", "algo diferente"
  
- FEEDBACK_POS: Acepta la recomendación explícitamente.
  Ejemplos: "me gusta", "perfecto", "vale", "de acuerdo", "la veo"
  
- FEEDBACK_NEG: La rechaza explícitamente.
  Ejemplos: "no quiero", "eso no", "no me interesa", "muy aburrido"
  
- SMALLTALK: Conversación trivial.
  Ejemplos: "hola", "gracias", "adiós", "buenos días"
  
- OTHER: Todo lo que no encaje claramente en las categorías anteriores.

No añadas texto fuera del JSON.
"""

EXTRACTION_PROMPT = """
Eres un modelo extractor de atributos para una Red Bayesiana.
Devuelve SOLO un JSON válido con los siguientes campos EXACTOS:

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
- No uses markdown ni bloques de código.
- Usa EXACTAMENTE los nombres de los campos indicados (respeta mayúsculas).

Convenciones de valores:

UserAge:
- "young" (18-35 años)
- "adult" (36-55 años)
- "senior" (56+ años)

UserGender:
- "male"
- "female"

HouseholdType:
- "single" (vive solo/a)
- "couple" (pareja)
- "family" (familia con hijos)

TimeOfDay:
- "morning" (07:00-12:00)
- "afternoon" (12:00-20:00)
- "night" (20:00-07:00)

DayType:
- "weekday" (lunes a viernes)
- "weekend" (sábado y domingo)

ProgramType:
- "movie" (película)
- "series" (serie de TV)
- "news" (noticias/informativos)
- "documentary" (documental)
- "entertainment" (entretenimiento/concursos)

ProgramGenre:
- "comedy" (comedia)
- "drama" (drama)
- "horror" (terror/suspense)
- "romance" (romántico)
- "news" (informativo)
- "documentary" (documental)
- "entertainment" (entretenimiento)
- "action" (acción)
- "thriller" (thriller)
- "sci-fi" (ciencia ficción)
- "fantasy" (fantasía)

ProgramDuration:
- "short" (menos de 30 minutos)
- "medium" (30-60 minutos)
- "long" (más de 60 minutos)

Notas importantes:
- Si el usuario menciona un tipo de programa pero no la duración, deja ProgramDuration en null.
- No deduzcas género ni duración salvo que se indiquen explícitamente.
- TimeOfDay y DayType se rellenarán automáticamente, pero si el usuario los menciona explícitamente, úsalos.
"""


# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def classify_intent(user_message: str) -> str:
    """
    Classify the user's intent using GPT-4
    
    Returns:
        Intent string: RECOMMEND, ALTERNATIVE, FEEDBACK_POS, FEEDBACK_NEG, SMALLTALK, OTHER
    """
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


def extract_attributes_llm(user_message: str) -> dict:
    """
    Extract BN attributes from user message using GPT-4
    
    Returns:
        Dictionary with BN attribute names and values
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())


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
        {"role": "system", "content": "STATE:\n" + json.dumps(state, indent=2, ensure_ascii=False)}
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
    Run BN inference to get Type and Genre recommendations
    
    Returns:
        Dictionary with ProgramType, ProgramGenre, and their rankings
    """
    attrs = recommend_by_genre(state)
    if not attrs:
        print(colorize("Not enough attributes for BN inference.", BN_LOG_COLOR))
        return {}

    # Infer ProgramType
    type_recs = recommend_type(attrs, model)
    chosen_type = type_recs[0][0]
    print(colorize(f"Type recommendations: {[t[0] for t in type_recs]}", BN_LOG_COLOR))

    # Infer ProgramGenre conditioned on ProgramType
    attrs_with_type = dict(attrs)
    attrs_with_type["ProgramType"] = chosen_type

    genre_recs = recommend_gender(attrs_with_type, model)
    chosen_genre = genre_recs[0][0]
    print(colorize(f"Genre recommendations: {[g[0] for g in genre_recs]}", BN_LOG_COLOR))

    print(colorize(
        f"BN inference: Type={chosen_type} | Genre={chosen_genre}",
        BN_LOG_COLOR
    ))

    return {
        "ProgramType": chosen_type,
        "ProgramGenre": chosen_genre,
        "type_ranking": [t[0] for t in type_recs],
        "genre_ranking": [g[0] for g in genre_recs],
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