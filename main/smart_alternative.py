"""
Smart alternative selection with genre rejection detection
Integrates with main.py to handle explicit genre rejections
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


GENRE_REJECTION_PROMPT = """
Analiza el mensaje del usuario para detectar si está rechazando explícitamente un GÉNERO específico.

Devuelve SOLO un JSON válido:
{
  "rejects_genre": true/false,
  "rejected_genre": "nombre del género rechazado o null",
  "reason": "breve explicación"
}

Ejemplos de RECHAZO DE GÉNERO (rejects_genre=true):
- "No me gusta el drama" → rejected_genre: "drama"
- "Nada de terror" → rejected_genre: "horror"
- "No quiero ver comedias" → rejected_genre: "comedy"
- "Algo que no sea romántico" → rejected_genre: "romance"
- "No me van los documentales" → rejected_genre: "documentary"

Ejemplos de RECHAZO DE CONTENIDO ESPECÍFICO (rejects_genre=false):
- "Esa no" → rejected_genre: null (rechaza el título, no el género)
- "Otra opción" → rejected_genre: null
- "No me convence" → rejected_genre: null
- "Muy larga" → rejected_genre: null

Géneros válidos:
- comedy, drama, horror, romance, action, thriller, sci-fi, fantasy
- documentary, news, entertainment

No añadas texto fuera del JSON.
"""


def detect_genre_rejection(user_message: str, current_genre: str) -> dict:
    """
    Detect if user is rejecting a specific genre
    
    Args:
        user_message: User's response
        current_genre: Currently recommended genre
        
    Returns:
        {
            "rejects_genre": bool,
            "rejected_genre": str or None,
            "reason": str
        }
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GENRE_REJECTION_PROMPT},
                {"role": "user", "content": f"Mensaje: '{user_message}'\nGénero actual: {current_genre}"}
            ],
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"Error detecting genre rejection: {e}")
        return {
            "rejects_genre": False,
            "rejected_genre": None,
            "reason": "Error in detection"
        }


def should_skip_to_next_genre(user_message: str, state: dict) -> tuple:
    """
    Determine if we should skip all content from current genre
    
    Args:
        user_message: User's response
        state: Current system state
        
    Returns:
        (should_skip: bool, rejected_genre: str or None)
    """
    current_genre = state.get("candidates", {}).get("ProgramGenre")
    
    if not current_genre:
        return False, None
    
    detection = detect_genre_rejection(user_message, current_genre)
    
    if detection["rejects_genre"]:
        rejected = detection["rejected_genre"]
        
        # Verify rejected genre matches current genre (or is similar)
        if rejected and (rejected == current_genre or rejected in current_genre or current_genre in rejected):
            return True, rejected
    
    return False, None


def get_next_different_genre(state: dict, content_fetcher, rejected_genre: str = None):
    """
    Get content from next genre in ranking, skipping rejected genre
    
    Args:
        state: Current system state
        content_fetcher: TMDB fetcher instance
        rejected_genre: Genre to skip (optional)
        
    Returns:
        bool: True if found alternative, False otherwise
    """
    from main import fetch_real_content, colorize, CONTENT_COLOR, WARNING_COLOR
    
    bn_result = state.get("candidates", {})
    genre_ranking = bn_result.get("genre_ranking", [])
    current_genre = bn_result.get("ProgramGenre")
    
    # Find current position in ranking
    try:
        current_idx = genre_ranking.index(current_genre)
    except ValueError:
        current_idx = 0
    
    # Try next genres in ranking
    for next_idx in range(current_idx + 1, len(genre_ranking)):
        next_genre = genre_ranking[next_idx]
        
        # Skip if this is the rejected genre
        if rejected_genre and next_genre == rejected_genre:
            print(colorize(f"   Skipping rejected genre: {next_genre}", WARNING_COLOR))
            continue
        
        print(colorize(f"   Trying next genre: {next_genre}", WARNING_COLOR))
        
        # Update BN result
        bn_result["ProgramGenre"] = next_genre
        
        # Fetch new content
        new_content = fetch_real_content(bn_result, content_fetcher, limit=10, fallback_to_alternatives=False)
        
        if new_content:
            state["real_content"] = new_content
            state["content_index"] = 0
            state["candidates"] = bn_result
            
            if state.get("last_recommendation"):
                state["last_recommendation"]["ProgramGenre"] = next_genre
                state["last_recommendation"]["content"] = new_content[0]
            
            print(colorize(f"Switched to genre: {next_genre}", CONTENT_COLOR))
            return True
    
    # If no genres left, try different type
    print(colorize("No more genres, trying different type...", WARNING_COLOR))
    type_ranking = bn_result.get("type_ranking", [])
    current_type = bn_result.get("ProgramType")
    
    try:
        current_type_idx = type_ranking.index(current_type)
        if current_type_idx + 1 < len(type_ranking):
            next_type = type_ranking[current_type_idx + 1]
            print(colorize(f"   Trying type: {next_type}", WARNING_COLOR))
            
            bn_result["ProgramType"] = next_type
            bn_result["ProgramGenre"] = genre_ranking[0] if genre_ranking else current_genre
            
            new_content = fetch_real_content(bn_result, content_fetcher, limit=10, fallback_to_alternatives=True)
            
            if new_content:
                state["real_content"] = new_content
                state["content_index"] = 0
                state["candidates"] = bn_result
                
                if state.get("last_recommendation"):
                    state["last_recommendation"]["ProgramType"] = next_type
                    state["last_recommendation"]["ProgramGenre"] = bn_result["ProgramGenre"]
                    state["last_recommendation"]["content"] = new_content[0]
                
                return True
    except (ValueError, IndexError):
        pass
    
    return False


# Example usage for testing
if __name__ == "__main__":
    # Test genre rejection detection
    test_cases = [
        ("No me gusta el drama", "drama"),
        ("Dame otra opción", "comedy"),
        ("Nada de terror", "horror"),
        ("Esa película no", "action"),
        ("No quiero ver comedias románticas", "romance"),
    ]
    
    for message, genre in test_cases:
        result = detect_genre_rejection(message, genre)
        print(f"\nMessage: '{message}'")
        print(f"Current genre: {genre}")
        print(f"Result: {result}")