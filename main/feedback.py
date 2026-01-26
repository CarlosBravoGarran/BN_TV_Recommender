from collections import defaultdict
import itertools
from pgmpy.factors.discrete import TabularCPD


# ============================================================================
# Initialize counts from existing CPDs
# ============================================================================

def initialize_cpt_counts(model, virtual_sample_size=100):
    """
    Convert model CPDs into initial in-memory counts
    using a virtual sample size for smoothing.
    """
    cpt_counts = {}

    for cpd in model.get_cpds():
        variable = cpd.variable
        parents = cpd.variables[1:]
        state_names = cpd.state_names

        counts = defaultdict(lambda: defaultdict(float))
        values = cpd.values

        for idx, state in enumerate(state_names[variable]):
            for parent_idx in itertools.product(
                *[range(len(state_names[p])) for p in parents]
            ):
                parent_state = tuple(
                    state_names[p][parent_idx[i]]
                    for i, p in enumerate(parents)
                )
                prob = values[(idx,) + parent_idx]
                counts[parent_state][state] = prob * virtual_sample_size

        cpt_counts[variable] = {
            "parents": parents,
            "state_names": state_names,
            "counts": counts
        }

    return cpt_counts


# ============================================================================
# Build CPD safely from counts
# ============================================================================

def build_cpd_from_counts(variable, cpt_info):
    parents = cpt_info["parents"]
    state_names = cpt_info["state_names"]
    counts = cpt_info["counts"]

    var_states = state_names[variable]
    parent_states = [state_names[p] for p in parents]
    parent_combinations = list(itertools.product(*parent_states))

    values = []

    for v in var_states:
        row = []
        for parent_state in parent_combinations:
            total = sum(counts[parent_state].values())
            prob = counts[parent_state][v] / total if total > 0 else 0.0
            row.append(prob)
        values.append(row)

    return TabularCPD(
        variable=variable,
        variable_card=len(var_states),
        values=values,
        evidence=parents,
        evidence_card=[len(ps) for ps in parent_states],
        state_names=state_names
    )


# ============================================================================
# Apply feedback (MINIMAL VERSION)
# ============================================================================

def apply_feedback(model, cpt_counts, state):
    """
    Update RecommendationAccepted | ProgramType
    using explicit user feedback.
    """

    feedback = state.get("user_feedback")
    if feedback not in ("accepted", "rejected"):
        return

    program_type = state.get("last_recommendation")
    if program_type is None:
        return

    accepted_state = "yes" if feedback == "accepted" else "no"

    cpt = cpt_counts.get("RecommendationAccepted")
    if cpt is None:
        return

    parent_state = (program_type,)

    # Safety check
    if parent_state not in cpt["counts"]:
        return

    # Increment count
    cpt["counts"][parent_state][accepted_state] += 500

    # Rebuild and replace CPD
    new_cpd = build_cpd_from_counts("RecommendationAccepted", cpt)
    model.remove_cpds("RecommendationAccepted")
    model.add_cpds(new_cpd)
