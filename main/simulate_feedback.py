"""
simulate_feedback.py

Development tool to simulate implicit user behaviour and test its effect
on the Bayesian Network CPDs.

Usage:
    python simulate_feedback.py
"""

import copy
import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import feedback as fb
import graph_builder


# ============================================================================
# Viewing → feedback translation
# ============================================================================

def viewing_to_feedback(percent_watched, times_watched=1, duration_minutes=None):
    """
    Translate viewing behaviour into feedback parameters.

    Args:
        percent_watched:  0-100 float, percentage of the content watched.
        times_watched:    How many times the user has watched this content.
        duration_minutes: Total duration of the content in minutes. If provided
                          and the actual watch time would be < 5 min the rule
                          takes priority over percent_watched.

    Returns:
        dict with keys 'user_feedback' and 'learning_rate', or None when no
        feedback should be applied (50-70 % range).
    """
    # Short-watch guard: < 5 minutes absolute → strong rejection
    if duration_minutes is not None:
        actual_minutes = (percent_watched / 100.0) * duration_minutes
        if actual_minutes < 5:
            lr = 100 * times_watched
            return {"user_feedback": "rejected", "learning_rate": lr}

    # Percentage-based rules
    if percent_watched >= 100:
        lr = 50 * times_watched
        return {"user_feedback": "accepted", "learning_rate": lr}

    if percent_watched > 70:
        lr = 25 * times_watched
        return {"user_feedback": "accepted", "learning_rate": lr}

    if percent_watched >= 50:
        # Neutral zone — no feedback
        return None

    # < 50 %
    lr = 50 * times_watched
    return {"user_feedback": "rejected", "learning_rate": lr}


# ============================================================================
# Session simulation
# ============================================================================

def simulate_session(
    model,
    cpt_counts,
    program_type,
    program_genre,
    percent_watched,
    times_watched=1,
    duration_minutes=None,
    context_attrs=None,
):
    """
    Build an artificial state and call apply_feedback.

    Args:
        model:            Loaded DiscreteBayesianNetwork.
        cpt_counts:       In-memory CPT counts (modified in place).
        program_type:     e.g. "movie", "series", "news" ...
        program_genre:    e.g. "comedy", "drama", "documentary" ...
        percent_watched:  0-100 float.
        times_watched:    Rewatch count (multiplies learning_rate).
        duration_minutes: Total duration in minutes (used for < 5 min rule).
        context_attrs:    Optional dict with BN context attributes:
                          UserAge, UserGender, HouseholdType, TimeOfDay, DayType.
                          Unknown / missing keys are treated as None (uniform
                          expansion by apply_feedback internally).

    Returns:
        The feedback params dict that was applied, or None if no feedback.
    """
    feedback_params = viewing_to_feedback(percent_watched, times_watched, duration_minutes)

    if feedback_params is None:
        print(
            f"[simulate] {program_type}/{program_genre} @ {percent_watched:.0f}% "
            f"-> neutral zone, no feedback applied."
        )
        return None

    attrs_bn = {
        "UserAge": None,
        "UserGender": None,
        "HouseholdType": None,
        "TimeOfDay": None,
        "DayType": None,
        "ProgramType": program_type,
        "ProgramGenre": program_genre,
        "ProgramDuration": None,
    }
    if context_attrs:
        attrs_bn.update(context_attrs)

    state = {
        "user_feedback": feedback_params["user_feedback"],
        "last_recommendation": {
            "ProgramType": program_type,
            "ProgramGenre": program_genre,
        },
        "atributes_bn": attrs_bn,
    }

    print(
        f"[simulate] {program_type}/{program_genre} @ {percent_watched:.0f}% "
        f"(x{times_watched}) -> {feedback_params['user_feedback']}, "
        f"lr={feedback_params['learning_rate']}"
    )

    fb.apply_feedback(model, cpt_counts, state, learning_rate=feedback_params["learning_rate"])
    return feedback_params


# ============================================================================
# CPD diff printer
# ============================================================================

def _counts_to_probs(cpt_info, variable):
    """
    Convert raw counts for *variable* into a probability table.

    Returns:
        dict mapping parent_state_tuple -> {child_state: probability}
    """
    state_names = cpt_info["state_names"]
    parents = cpt_info["parents"]
    var_states = state_names[variable]

    if parents:
        parent_combos = list(itertools.product(*[state_names[p] for p in parents]))
    else:
        parent_combos = [()]

    probs = {}
    for parent_state in parent_combos:
        total = sum(cpt_info["counts"][parent_state].values())
        if total > 0:
            probs[parent_state] = {
                s: cpt_info["counts"][parent_state][s] / total for s in var_states
            }
        else:
            n = len(var_states)
            probs[parent_state] = {s: 1.0 / n for s in var_states}
    return probs


def print_cpd_diff(cpt_counts_before, cpt_counts_after, variable):
    """
    Print a before/after comparison of the probability distributions for
    *variable*. Only rows where at least one probability changed are shown.

    Args:
        cpt_counts_before: Deep copy of cpt_counts taken before apply_feedback.
        cpt_counts_after:  cpt_counts after apply_feedback.
        variable:          BN variable name, e.g. "ProgramType" or "ProgramGenre".
    """
    if variable not in cpt_counts_before or variable not in cpt_counts_after:
        print(f"[diff] Variable '{variable}' not found in cpt_counts.")
        return

    probs_before = _counts_to_probs(cpt_counts_before[variable], variable)
    probs_after  = _counts_to_probs(cpt_counts_after[variable],  variable)

    parents    = cpt_counts_after[variable]["parents"]
    var_states = cpt_counts_after[variable]["state_names"][variable]

    changed_rows = [
        parent_state
        for parent_state, after_dist in probs_after.items()
        if any(
            abs(after_dist.get(s, 0) - probs_before.get(parent_state, {}).get(s, 0)) > 1e-8
            for s in var_states
        )
    ]

    if not changed_rows:
        print(f"[diff] No changes detected for '{variable}'.")
        return

    col_w = 14
    state_header = "  ".join(f"{s:<{col_w}}" for s in var_states)
    separator = "-" * (col_w * len(var_states) + 2 * len(var_states) + 40)

    print(f"\n{'='*60}")
    print(f" CPD diff for: {variable}")
    if parents:
        print(f" Parents: {', '.join(parents)}")
    print(f"{'='*60}")
    print(f"  {'Parent state':<38}  {state_header}")
    print(separator)

    for parent_state in changed_rows:
        before_dist = probs_before.get(parent_state, {})
        after_dist  = probs_after[parent_state]

        parent_label = str(parent_state) if parent_state else "(no parents)"

        before_str = "  ".join(f"{before_dist.get(s, 0):.4f}        " for s in var_states)
        after_str  = "  ".join(f"{after_dist.get(s, 0):.4f}        " for s in var_states)
        delta_str  = "  ".join(
            f"{after_dist.get(s, 0) - before_dist.get(s, 0):+.4f}        " for s in var_states
        )

        print(f"  {parent_label}")
        print(f"    before : {before_str}")
        print(f"    after  : {after_str}")
        print(f"    delta  : {delta_str}")
        print()


# ============================================================================
# Deep-copy helper (handles defaultdict-based cpt_counts)
# ============================================================================

def snapshot_counts(cpt_counts):
    """Return a deep copy of cpt_counts safe for diffing."""
    return copy.deepcopy(cpt_counts)


# ============================================================================
# Main: demonstration examples
# ============================================================================

if __name__ == "__main__":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "model.pkl")

    print("Loading model ...")
    model = graph_builder.load_model(MODEL_PATH)

    print("Initialising CPT counts ...")
    cpt_counts = fb.initialize_cpt_counts(model)

    # -----------------------------------------------------------------------
    # Example 1 – Strong acceptance: watched 100%, senior couple, weekday afternoon
    # -----------------------------------------------------------------------
    context_couple_afternoon = {
        "DayType": "weekday",
        "TimeOfDay": "afternoon",
        "HouseholdType": "couple",
        "UserAge": "senior",
        "UserGender": "female",
    }

    before_1 = snapshot_counts(cpt_counts)

    simulate_session(
        model, cpt_counts,
        program_type="series",
        program_genre="drama",
        percent_watched=100,
        context_attrs=context_couple_afternoon,
    )

    print_cpd_diff(before_1, cpt_counts, "ProgramType")
    print_cpd_diff(before_1, cpt_counts, "ProgramGenre")

    # -----------------------------------------------------------------------
    # Example 2 – Neutral zone (60%): no feedback should be applied
    # -----------------------------------------------------------------------
    before_2 = snapshot_counts(cpt_counts)

    simulate_session(
        model, cpt_counts,
        program_type="movie",
        program_genre="comedy",
        percent_watched=60,
    )

    print_cpd_diff(before_2, cpt_counts, "ProgramType")  # Expects: no changes

    # -----------------------------------------------------------------------
    # Example 3 – Rejection: watched only 30%, weekend morning
    # -----------------------------------------------------------------------
    before_3 = snapshot_counts(cpt_counts)

    simulate_session(
        model, cpt_counts,
        program_type="news",
        program_genre="news",
        percent_watched=30,
        context_attrs={"DayType": "weekend", "TimeOfDay": "morning"},
    )

    print_cpd_diff(before_3, cpt_counts, "ProgramType")
    print_cpd_diff(before_3, cpt_counts, "ProgramGenre")

    # -----------------------------------------------------------------------
    # Example 4 – Short watch < 5 min: strong rejection despite high percent
    #   (watched 80% of a 4-minute clip → only 3.2 min → < 5 min rule fires)
    # -----------------------------------------------------------------------
    before_4 = snapshot_counts(cpt_counts)

    simulate_session(
        model, cpt_counts,
        program_type="documentary",
        program_genre="documentary",
        percent_watched=80,
        duration_minutes=4,
    )

    print_cpd_diff(before_4, cpt_counts, "ProgramType")

    # -----------------------------------------------------------------------
    # Example 5 – Rewatch x3: learning_rate tripled (strong positive signal)
    # -----------------------------------------------------------------------
    before_5 = snapshot_counts(cpt_counts)

    simulate_session(
        model, cpt_counts,
        program_type="movie",
        program_genre="romance",
        percent_watched=100,
        times_watched=3,
        context_attrs=context_couple_afternoon,
    )

    print_cpd_diff(before_5, cpt_counts, "ProgramType")
    print_cpd_diff(before_5, cpt_counts, "ProgramGenre")
