import json
import os
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from bn_recommender import recomendar_generos_bn

# Logging
logging.getLogger("httpx").setLevel(logging.WARNING)

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
- contexto del usuario (hora, día, género)
- candidatos del recomendador
- la última recomendación del turno anterior
- feedback del usuario ("accepted" o "rejected")
- el historial reciente de recomendaciones y respuestas

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
8. Responder SIEMPRE con un JSON con los campos:

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
Eres un modelo extractor. Devuelve SOLO un JSON válido con estos campos:

{
 "Hora": ...,
 "DiaSemana": ...,
 "GeneroUsuario": ...,
 "EdadUsuario": ...,
 "DuracionPrograma": ...,
 "TipoEmision": ...,
 "InteresPrevio": ...,
 "GeneroPrograma": ...,
 "PopularidadPrograma": ...,
 "Satisfaccion": ...,
 "Recomendado": ...
}

Reglas:
- Si no se menciona algo, usa null.
- No añadas texto antes ni después del JSON.
- No incluyas explicaciones. 
- No uses markdown. 
- Para la hora y día coge el los actuales si no se mencionan. 
- La edad clasifícala en rangos: joven (18-35), adulto (36-55), mayor (56+). 
- Duración del programa en minutos: corta (<30), media (30-60), larga (60+) 
- Si indican el tipo de programa (serie, película, documental, etc) úsalo para la duración del programa. 
- GéneroUsuario: "hombre" o "mujer" - Hora: "mañana", "tarde", "noche" - Día: "laboral" o "fin_semana" 
- TipoEmision: "bajo_demanda", "diferido", "directo"
"""

# Intent classifier function

def clasificar_intencion(mensaje_usuario: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": mensaje_usuario}
        ],
        temperature=0
    )
    data = json.loads(resp.choices[0].message.content)
    return data["intent"]

# Attribute extraction

def extraer_atributos_llm(mensaje_usuario: str) -> dict:

    respuesta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": mensaje_usuario}
        ],
        temperature=0
    )

    return json.loads(respuesta.choices[0].message.content.strip())

# BN inference

def recomendar_por_genero(state: dict) -> dict:

    attrs = {}
    if isinstance(state.get("context"), dict) and "atributos_bn" in state["context"]:
        attrs = state["context"].get("atributos_bn") or {}

    filtrados = {k: v for k, v in attrs.items() if v not in (None, "", "null")}
    print("Atributos no nulos para BN:", filtrados)

    return filtrados

def inferir_con_bn(state: dict) -> list:

    attrs = recomendar_por_genero(state)
    if not attrs:
        print("No hay atributos suficientes para inferencia BN.")
        return []

    recomendaciones = recomendar_generos_bn(attrs)
    ordenados = [g for g, _ in recomendaciones]

    print("Recomendaciones BN:", ordenados)
    return ordenados

# Conversational LLM

def conversar(mensaje_usuario, state, historial=None):

    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "STATE:\n" + json.dumps(state, indent=2)}
    ]

    if historial:
        mensajes.extend(historial)

    mensajes.append({"role": "user", "content": mensaje_usuario})

    respuesta = client.chat.completions.create(
        model="gpt-4o",
        messages=mensajes,
        temperature=0.5
    )

    return respuesta.choices[0].message.content


# MAIN LOOP

if __name__ == "__main__":
    
    historial = []
    states_log = []

    state = {
        "context": {
            "hour": datetime.now().hour,
            "day": datetime.now().strftime("%A"),
        },
        "candidates": [],
        "last_recommendation": None,
        "user_feedback": None,
        "interaction_history": []
    }

    print("Asistente de TV. Escribe 'salir' para terminar.\n")

    while True:
        mensaje = input("Usuario: ")

        if mensaje.lower().strip() == "salir":
            break

        # 1) Clasificar intención
        intent = clasificar_intencion(mensaje)
        print("Intent detectado:", intent)

        # 2) Lógica según intención

        if intent == "RECOMMEND":
            atributos = extraer_atributos_llm(mensaje)
            state["context"]["atributos_bn"] = atributos
            state["candidates"] = inferir_con_bn(state)

        elif intent == "ALTERNATIVE":
            state["user_feedback"] = "rejected"

        elif intent == "FEEDBACK_POS":
            state["user_feedback"] = "accepted"

        elif intent == "FEEDBACK_NEG":
            state["user_feedback"] = "rejected"

        elif intent == "SMALLTALK":
            pass  # no tocar BN

        elif intent == "OTHER":
            pass

        # Guardar state
        states_log.append(json.loads(json.dumps(state)))

        # 3) Conversación final

        raw_response = conversar(mensaje, state, historial)

        try:
            response = json.loads(raw_response)
        except:
            print("ERROR JSON:", raw_response)
            continue

        action = response.get("action")
        message = response.get("message")
        item = response.get("item")

        print("Asistente:", message)

        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": message})

        if item:
            state["last_recommendation"] = item

        if action == "ALTERNATIVE":
            state["user_feedback"] = "rejected"
        elif action == "RECOMMEND":
            state["user_feedback"] = None

        if item:
            state["interaction_history"].append({
                "item": item,
                "feedback": state["user_feedback"]
            })

    # Guardado final de states

    save_path = Path(__file__).parent / "states.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(states_log, f, indent=2, ensure_ascii=False)

    print(f"Todos los STATES guardados en {save_path}")
