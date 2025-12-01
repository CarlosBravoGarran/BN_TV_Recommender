import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import logging

from bn_recommender import recomendar_generos_bn

import google.generativeai as genai

logging.getLogger("httpx").setLevel(logging.WARNING)

env_path = ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

MODEL_CHAT = genai.GenerativeModel("gemini-2.5-pro")
MODEL_EXTRACTION = genai.GenerativeModel("gemini-2.5-pro")

SYSTEM_PROMPT = """ (… tu prompt original …) """

EXTRACTION_PROMPT = """ (… tu extraction prompt original …) """


def extraer_atributos_llm(mensaje_usuario: str) -> dict:
    prompt = EXTRACTION_PROMPT + "\n\n" + mensaje_usuario
    respuesta = MODEL_EXTRACTION.generate_content(prompt)
    crudo = respuesta.text.strip()
    atributos = json.loads(crudo)
    return atributos


def conversar(mensaje_usuario, state, historial=None):
    if historial is None:
        historial = []

    texto_historial = ""
    for h in historial:
        if h["role"] == "user":
            texto_historial += f"[USUARIO]\n{h['content']}\n"
        else:
            texto_historial += f"[ASISTENTE]\n{h['content']}\n"

    prompt = (
        "[SYSTEM]\n" + SYSTEM_PROMPT +
        "\n\n[STATE]\n" + json.dumps(state, indent=2) +
        "\n\n[HISTORIAL]\n" + texto_historial +
        "\n\n[USER]\n" + mensaje_usuario
    )

    respuesta = MODEL_CHAT.generate_content(prompt)
    return respuesta.text


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

    save_path = Path(__file__).parent / "states.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(states_log, f, indent=2, ensure_ascii=False)

    print(f"Todos los STATES guardados en states.json")
