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


def main():
    history = []
    states_log = []

    state = {
        "atributes_bn": {},
        "candidates": {},
        "last_recommendation": None,
        "user_feedback": None,
    }

    print("TV Assistant. Type 'exit' to quit.\n")

    model = load_model("main/outputs/model.pkl")

    # Initialize CPT counts
    cpt_counts = initialize_cpt_counts(model, virtual_sample_size=100)

    # Load previous counts if they exist
    counts_path = Path(__file__).parent / "outputs/cpt_counts.json"
    if counts_path.exists():
        try:
            cpt_counts = load_cpt_counts(counts_path)
        except Exception as e:
            print(f"Could not load previous counts: {e}")

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
            state["last_recommendation"] = {
                "ProgramType": bn_result["ProgramType"],
                "ProgramGenre": bn_result["ProgramGenre"],
            }
            # Reset feedback for new recommendation
            state["user_feedback"] = None

        elif intent == "ALTERNATIVE":
            state["user_feedback"] = "rejected"
            # Apply feedback
            apply_feedback(model, cpt_counts, state, learning_rate=50)

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

        states_log.append(json.loads(json.dumps(state)))

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

    save_path = Path(__file__).parent / "states.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(states_log, f, indent=2, ensure_ascii=False)

    print("All STATES saved to states.json")


if __name__ == "__main__":
    main()
