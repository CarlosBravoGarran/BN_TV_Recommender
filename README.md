# BN TV Recommender

Sistema de recomendación conversacional de contenido televisivo que combina una **red bayesiana discreta** como motor de inferencia probabilístico con un **agente LLM** como interfaz en lenguaje natural. Desarrollado como Trabajo Fin de Grado (Doble Grado en Ingeniería Informática y Administración de Empresas, Universidad Carlos III de Madrid) en colaboración con **MasOrange**, como prueba de concepto orientada a su plataforma de streaming **OrangeTV**.

## Motivación

Los sistemas de recomendación dominantes (filtrado colaborativo, aprendizaje profundo) dependen de grandes volúmenes de historial de interacción, no ofrecen interfaces en lenguaje natural y sus decisiones son opacas. En el dominio de IPTV esto se agrava: el consumo televisivo es muy contextual (franja horaria, tipo de día, composición del hogar) y las plataformas atienden a perfiles de usuario muy diversos, muchos sin historial previo.

Este proyecto explora una arquitectura híbrida que responde a esas tres carencias a la vez:

- **Arranque en frío**: no necesita historial de interacciones para recomendar, gracias a una base de datos sintética que aporta *priors* realistas.
- **Adaptación en tiempo real**: las preferencias del usuario se incorporan al instante mediante feedback, sin reentrenar el modelo.
- **Interacción en lenguaje natural**: el usuario conversa con el sistema en español en lugar de usar filtros o catálogos.

## Arquitectura

El sistema se organiza en tres capas con responsabilidades independientes:

```
┌─────────────────────────────────────────────────┐
│  Interfaz de usuario (navegador + servidor Flask)│
└───────────────────────┬───────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────┐
│  Agente LLM                                        │
│  Clasificación de intención → Extracción de        │
│  atributos → Respuesta conversacional               │
│  (estado de conversación compartido)                │
└───────────────────────┬───────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────┐
│  Motor de inferencia y recuperación                │
│  Red bayesiana (inferencia/CPDs) · Feedback online  │
│  (actualización de CPDs) · Cliente TMDB (títulos)   │
└─────────────────────────────────────────────────────┘
```

- **Capa de interfaz**: aplicación web ligera servida por Flask que recoge los mensajes del usuario y muestra las recomendaciones.
- **Capa del agente LLM**: no decide qué recomendar; traduce el lenguaje natural del usuario a estructuras de datos (y viceversa) en tres llamadas independientes:
  1. **Clasificación de intención** (GPT-4o): determina si el usuario pide una recomendación, valora positiva/negativamente lo recibido, pide una alternativa o mantiene una conversación trivial.
  2. **Extracción de atributos** (Claude Sonnet 4.5): extrae tipo de contenido, género, duración, perfil demográfico u otras preferencias mencionadas, dejando como evidencia ausente lo no mencionado.
  3. **Generación de respuesta conversacional**: redacta el mensaje final en español.
- **Capa del motor de inferencia y recuperación**:
  - **Red bayesiana**: modela las relaciones entre el perfil del usuario, el contexto temporal (franja horaria, tipo de día, determinados automáticamente en cada consulta) y el tipo/género de contenido recomendado. La estructura se aprende con `HillClimbingSearch` y la métrica **BDeu** (ESS=100), con restricciones de *whitelist*/*blacklist* derivadas de conocimiento experto del dominio.
  - **Feedback online**: actualiza las CPDs en tiempo real mediante un mecanismo de **pseudoconteos** cuando el usuario valora una recomendación, sin necesidad de reentrenar el modelo.
  - **Cliente TMDB**: consulta la API de [TMDB](https://www.themoviedb.org/) para traducir el tipo/género inferido en títulos reales.
  - **Simulación de feedback**: módulo exclusivo de evaluación que simula feedback implícito de visionado (no forma parte del flujo con el usuario final, al no existir aún integración con una plataforma de reproducción real).

## Estructura del repositorio

```
main/
├── api.py                 # Servidor Flask: expone el pipeline al frontend
├── main.py                # Runner de consola para probar el pipeline completo
├── bn_builder.py           # Construcción/aprendizaje de la estructura de la red bayesiana
├── bn_recommender.py       # Lógica de recomendación sobre el modelo
├── inference.py            # Inferencia sobre la red bayesiana
├── feedback.py              # Actualización de CPDs mediante pseudoconteos
├── LLM_agent.py             # Clasificación de intención, extracción de atributos, respuesta
├── content_fetcher.py       # Cliente de la API de TMDB
├── smart_alternative.py     # Lógica de alternativas cuando se rechaza una recomendación
├── dataset_gen.py           # Generación de la base de datos sintética de perfiles
├── simulate_feedback.py     # Simulación de feedback implícito para evaluación
├── LLM_evaluation.ipynb     # Evaluación cuantitativa del módulo LLM
├── bn_test.ipynb            # Pruebas y validación de la red bayesiana
└── output/                  # Modelo entrenado, CPDs, perfiles de usuario, resultados

frontend/                    # Interfaz web (HTML/CSS/JS) servida por Flask
score_testing/                # Comparativa de estructuras de red con distinto número de nodos
test_classifier/               # Prototipos de clasificador bayesiano (Hill Climbing vs. Variable Elimination)
```

## Puesta en marcha

1. Crear y activar un entorno virtual, e instalar dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configurar las claves de API necesarias en un archivo `.env` en la raíz del proyecto:

   ```
   OPENAI_API_KEY=...
   TMDB_API_KEY=...
   ```

3. Lanzar el servidor:

   ```bash
   cd main
   python api.py
   ```

   El servidor sirve la interfaz web y expone la API en `http://localhost:5000`.

   Alternativamente, `python main.py` ejecuta el pipeline completo por consola sin interfaz web.

## Evaluación

El módulo LLM se evalúa sobre conjuntos de test controlados: la clasificación de intención (GPT-4o) alcanza un **93,3 % de exactitud** (F1 ponderado 0,93) sobre 300 casos, y la extracción de atributos (Claude Sonnet 4.5) obtiene un **80,5 % de casos perfectos** sobre 200 casos, con precisión por campo superior al 93 % en todos los atributos. La red bayesiana se valida mediante análisis lógico de su estructura, y el sistema de feedback mediante simulación de iteraciones sobre distintos perfiles de usuario. Los detalles completos están en `main/LLM_evaluation.ipynb` y `main/bn_test.ipynb`.

## Alcance y limitaciones

Este es un prototipo académico y no incluye despliegue en producción: no hay integración real con OrangeTV, autenticación de usuarios ni infraestructura escalable. La base de datos de entrenamiento es sintética (construida a partir de estadísticas de audiencia del mercado español, no de datos reales de usuarios de la plataforma), y el feedback implícito se simula ante la ausencia de una plataforma de reproducción real. La arquitectura prioriza la interpretabilidad y la adaptación con pocos datos frente a la precisión de ranking de los modelos de aprendizaje profundo, que requieren historiales de interacción extensos.

## Licencia

Este proyecto está sujeto a la licencia [Creative Commons Reconocimiento - No Comercial - Sin Obra Derivada](LICENSE).
