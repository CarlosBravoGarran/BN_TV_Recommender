import json
from pathlib import Path

from feedback import initialize_cpt_counts, apply_feedback, load_cpt_counts, save_cpt_counts
from graph_builder import load_model
from LLM_agent import (
    ACTION_COLORS,
    INTENT_COLORS,
    classify_intent,
    colorize,
    converse,
    extract_attributes_llm,
    get_time_daytype,
    infer_with_bn,
)
from content_fetch import TMDBContentFetcher

# Nuevos colores para content
CONTENT_COLOR = "\033[96m"
WARNING_COLOR = "\033[93m"


def fetch_real_content(
    bn_result: dict, 
    content_fetcher: TMDBContentFetcher, 
    limit: int = 10,
    fallback_to_alternatives: bool = True
) -> list:
    """
    Fetch real content from TMDB based on BN recommendation
    
    Returns:
        List of content items (may be empty)
    """
    program_type = bn_result.get("ProgramType")
    program_genre = bn_result.get("ProgramGenre")
    
    if not program_type or not program_genre:
        print(colorize("⚠️  Missing Type or Genre for content fetch", WARNING_COLOR))
        return []
    
    print(colorize(f"🎬 Fetching content: Type={program_type}, Genre={program_genre}", CONTENT_COLOR))
    
    try:
        content = content_fetcher.get_content_by_recommendation(
            program_type=program_type,
            program_genre=program_genre,
            limit=limit,
            language="es-ES"
        )
        
        if content:
            print(colorize(f"✅ Found {len(content)} items:", CONTENT_COLOR))
            for i, item in enumerate(content[:3], 1):
                rating = item.get('vote_average', 0)
                print(colorize(f"   {i}. {item['title']} ({rating}/10)", CONTENT_COLOR))
            return content
        
        # No content found - try alternatives
        if fallback_to_alternatives:
            print(colorize(f"⚠️  No content for {program_genre}, trying alternatives...", WARNING_COLOR))
            
            # Try alternative genres from BN ranking
            genre_ranking = bn_result.get("genre_ranking", [])
            for alt_genre in genre_ranking[1:4]:  # Try top 3 alternatives
                print(colorize(f"   Trying alternative: {alt_genre}", WARNING_COLOR))
                
                alt_content = content_fetcher.get_content_by_recommendation(
                    program_type=program_type,
                    program_genre=alt_genre,
                    limit=limit,
                    language="es-ES"
                )
                
                if alt_content:
                    print(colorize(f"✅ Found content with alternative genre: {alt_genre}", CONTENT_COLOR))
                    # Update the genre in bn_result
                    bn_result["ProgramGenre"] = alt_genre
                    return alt_content
            
            # Still no content - try trending as last resort
            print(colorize("   Trying trending content as fallback...", WARNING_COLOR))
            trending = content_fetcher.get_trending(
                media_type="movie" if program_type == "movie" else "tv",
                time_window="week",
                language="es-ES"
            )
            
            if trending:
                print(colorize(f"✅ Using {len(trending)} trending items as fallback", CONTENT_COLOR))
                return trending[:limit]
        
        print(colorize("❌ No content available for this recommendation", WARNING_COLOR))
        return []
    
    except Exception as e:
        print(colorize(f"❌ Error fetching content: {e}", WARNING_COLOR))
        return []


def try_next_alternative(state: dict, content_fetcher: TMDBContentFetcher) -> bool:
    """
    Try to get the next alternative content.
    Returns True if successful, False if no more alternatives available.
    """
    # First, try next item in current list
    state["content_index"] = state.get("content_index", 0) + 1
    real_content = state.get("real_content", [])
    
    if real_content and state["content_index"] < len(real_content):
        # We have more items in the current list
        next_content = real_content[state["content_index"]]
        if state.get("last_recommendation"):
            state["last_recommendation"]["content"] = next_content
        print(colorize(f"📋 Moving to item {state['content_index'] + 1}/{len(real_content)}", CONTENT_COLOR))
        return True
    
    # No more in current list - try next genre from BN
    print(colorize("🔄 End of content list, trying next genre...", WARNING_COLOR))
    
    bn_result = state.get("candidates", {})
    genre_ranking = bn_result.get("genre_ranking", [])
    current_genre = bn_result.get("ProgramGenre")
    
    # Find next genre in ranking
    try:
        current_idx = genre_ranking.index(current_genre)
        if current_idx + 1 < len(genre_ranking):
            next_genre = genre_ranking[current_idx + 1]
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
                
                return True
    except (ValueError, IndexError):
        pass
    
    # Try different type as last resort
    print(colorize("🔄 Trying different program type...", WARNING_COLOR))
    type_ranking = bn_result.get("type_ranking", [])
    current_type = bn_result.get("ProgramType")
    
    try:
        current_idx = type_ranking.index(current_type)
        if current_idx + 1 < len(type_ranking):
            next_type = type_ranking[current_idx + 1]
            print(colorize(f"   Trying type: {next_type}", WARNING_COLOR))
            
            # Reset to first genre for new type
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
    
    print(colorize("❌ No more alternatives available", WARNING_COLOR))
    return False


def main():
    history = []
    states_log = []

    state = {
        "atributes_bn": {},
        "candidates": {},
        "last_recommendation": None,
        "user_feedback": None,
        "real_content": [],  # Real content from TMDB
        "content_index": 0,   # Track which content we're recommending
        "content_available": False  # Flag for content availability
    }

    print("🎬 TV Assistant with Real Content. Type 'exit' to quit.\n")

    model = load_model("main/output/model.pkl")

    # Initialize CPT counts
    cpt_counts = initialize_cpt_counts(model, virtual_sample_size=100)

    # Load previous counts if they exist
    counts_path = Path(__file__).parent / "output/cpt_counts.json"
    if counts_path.exists():
        try:
            cpt_counts = load_cpt_counts(counts_path)
            print("📂 Loaded previous learning data\n")
        except Exception as e:
            print(f"⚠️  Could not load previous counts: {e}\n")

    # Initialize TMDB content fetcher
    try:
        content_fetcher = TMDBContentFetcher()
        print("✅ TMDB API connected\n")
    except Exception as e:
        print(colorize(f"⚠️  Could not connect to TMDB: {e}", WARNING_COLOR))
        print(colorize("   System will work in genre-only mode", WARNING_COLOR))
        print(colorize("   Set TMDB_API_KEY in .env to enable real content\n", WARNING_COLOR))
        content_fetcher = None

    while True:
        mensaje = input("User: ")

        if mensaje.lower().strip() == "exit":
            break

        intent = classify_intent(mensaje)
        intent_msg = f"Detected intent: {intent}"
        print(colorize(intent_msg, INTENT_COLORS.get(intent, "")))

        if intent == "RECOMMEND":
            atributes = extract_attributes_llm(mensaje)
            time_of_day, day_type = get_time_daytype()

            atributes["TimeOfDay"] = time_of_day
            atributes["DayType"] = day_type

            state["atributes_bn"] = atributes
            bn_result = infer_with_bn(state, model)

            state["candidates"] = bn_result
            
            # Fetch real content from TMDB (if available)
            if content_fetcher:
                real_content = fetch_real_content(bn_result, content_fetcher, limit=10)
                state["real_content"] = real_content
                state["content_index"] = 0
                state["content_available"] = len(real_content) > 0
                
                # Set recommendation
                if real_content:
                    state["last_recommendation"] = {
                        "ProgramType": bn_result["ProgramType"],
                        "ProgramGenre": bn_result["ProgramGenre"],
                        "content": real_content[0]
                    }
                else:
                    # No real content, but we can still recommend by genre
                    state["last_recommendation"] = {
                        "ProgramType": bn_result["ProgramType"],
                        "ProgramGenre": bn_result["ProgramGenre"]
                    }
                    print(colorize("ℹ️  No specific titles available, will recommend by genre", WARNING_COLOR))
            else:
                # No TMDB connection - genre-only mode
                state["real_content"] = []
                state["content_available"] = False
                state["last_recommendation"] = {
                    "ProgramType": bn_result["ProgramType"],
                    "ProgramGenre": bn_result["ProgramGenre"]
                }
            
            # Reset feedback for new recommendation
            state["user_feedback"] = None

        elif intent == "ALTERNATIVE":
            state["user_feedback"] = "rejected"
            # Apply feedback
            apply_feedback(model, cpt_counts, state, learning_rate=50)
            
            # Try to get next alternative
            if content_fetcher and state.get("content_available"):
                has_alternative = try_next_alternative(state, content_fetcher)
                state["content_available"] = has_alternative
                
                if not has_alternative:
                    print(colorize("ℹ️  No more specific titles, switching to genre recommendations", WARNING_COLOR))
            else:
                # Genre-only mode - offer next genre from ranking
                bn_result = state.get("candidates", {})
                genre_ranking = bn_result.get("genre_ranking", [])
                current_genre = bn_result.get("ProgramGenre")
                
                try:
                    current_idx = genre_ranking.index(current_genre)
                    if current_idx + 1 < len(genre_ranking):
                        next_genre = genre_ranking[current_idx + 1]
                        bn_result["ProgramGenre"] = next_genre
                        state["candidates"] = bn_result
                        if state.get("last_recommendation"):
                            state["last_recommendation"]["ProgramGenre"] = next_genre
                        print(colorize(f"📋 Next genre: {next_genre}", CONTENT_COLOR))
                except (ValueError, IndexError):
                    print(colorize("ℹ️  No more genre alternatives", WARNING_COLOR))

        elif intent == "FEEDBACK_POS":
            state["user_feedback"] = "accepted"
            # Apply feedback
            apply_feedback(model, cpt_counts, state, learning_rate=50)

        elif intent == "FEEDBACK_NEG":
            state["user_feedback"] = "rejected"
            # Apply feedback
            apply_feedback(model, cpt_counts, state, learning_rate=50)

        elif intent in ("SMALLTALK", "OTHER"):
            pass

        states_log.append(json.loads(json.dumps(state, default=str)))

        raw_response = converse(mensaje, state, history)

        try:
            response = json.loads(raw_response)
        except Exception:
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

    # Save CPT counts
    save_cpt_counts(cpt_counts, counts_path)
    print(f"\n💾 Learning data saved")

    save_path = Path(__file__).parent / "output/states.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(states_log, f, indent=2, ensure_ascii=False)

    print("📝 All states saved to states.json")


if __name__ == "__main__":
    main()