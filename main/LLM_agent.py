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

# Cargar .env
env_path = ".env"
load_dotenv(env_path)

# Inicializar cliente OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not defined in .env file")

client = OpenAI(api_key=api_key)

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

9. Es muy importante que recojas el feedback del usuario para ver si ha aceptado o rechazado la recomendación previa.
"""

EXTRACTION_PROMPT = """
Eres un modelo extractor. Debes devolver UNICAMENTE un JSON válido con los siguientes campos.
Si no puedes inferir alguno, usa null.

Campos:
{
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
- No añadas texto antes ni después del JSON.
- No incluyas explicaciones.
- No uses markdown.
- Para la hora y día coge el los actuales si no se mencionan.
- La edad clasifícala en rangos: joven (18-35), adulto (36-55), mayor (56+).
- Duración del programa en minutos: corto (<30), medio (30-60), largo (60+)
- Si indican el tipo de programa (serie, película, documental, etc) úsalo para la duración del programa.
"""

def conversar(mensaje_usuario, state, historial=None):
    if historial is None:
        historial = []

    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "STATE:\n" + json.dumps(state, indent=2)}
    ]

    mensajes.extend(historial)
    mensajes.append({"role": "user", "content": mensaje_usuario})

    respuesta = client.chat.completions.create(
        model="gpt-4o",
        messages=mensajes,
        temperature=0.5
    )

    content = respuesta.choices[0].message.content
    return content


def extraer_atributos_llm(mensaje_usuario: str) -> dict:

    respuesta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": mensaje_usuario}
        ],
        temperature=0
    )

    crudo = respuesta.choices[0].message.content.strip()
    atributos = json.loads(crudo)
    return atributos



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

        atributos = extraer_atributos_llm(mensaje)
        state["context"]["atributos_bn"] = atributos

        states_log.append(json.loads(json.dumps(state)))


        if mensaje.lower().strip() == "salir":
            break

        raw_response = conversar(mensaje, state, historial)

        try:
            response = json.loads(raw_response)
        except:
            print("Error en el formato del modelo:", raw_response)
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
    
    # Guardar todos los STATES en un archivo JSON
    save_path = Path(__file__).parent / "states.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(states_log, f, indent=2, ensure_ascii=False)

    print(f"Todos los STATES guardados en states.json")
