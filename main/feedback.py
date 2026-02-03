# feedback.py

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
    """
    Rebuild a TabularCPD from the counts dictionary.
    """
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
# Apply feedback (ProgramType, ProgramGenre)
# ============================================================================

def apply_feedback(model, cpt_counts, state, learning_rate=50):
    """
    Update CPDs based on user feedback for Type and Genre recommendations.
    
    Args:
        model: The Bayesian Network model
        cpt_counts: Dictionary with count data for each CPD
        state: Current state containing last_recommendation and user_feedback
        learning_rate: How much to increment counts (higher = faster learning)
    
    Strategy:
        - If accepted: reinforce the recommended Type and Genre
        - If rejected: penalize the recommended pair, boost alternatives
    """
    
    feedback = state.get("user_feedback")
    if feedback not in ("accepted", "rejected"):
        return
    
    last_rec = state.get("last_recommendation")
    if not last_rec:
        return
    
    program_type = last_rec.get("ProgramType")
    program_genre = last_rec.get("ProgramGenre")
    
    if not program_type or not program_genre:
        return
    
    # Get contextual attributes for conditioning
    attrs = state.get("atributes_bn", {})
    
    print(f"\n Applying feedback: {feedback} for Type={program_type}, Genre={program_genre}")
    
    # ========================================
    # 1. Update ProgramType CPD
    # ========================================
    if "ProgramType" in cpt_counts:
        update_program_type_cpd(
            model, cpt_counts, program_type, attrs, feedback, learning_rate
        )
    
    # ========================================
    # 2. Update ProgramGenre CPD
    # ========================================
    if "ProgramGenre" in cpt_counts:
        update_program_genre_cpd(
            model, cpt_counts, program_type, program_genre, attrs, feedback, learning_rate
        )


def update_program_type_cpd(model, cpt_counts, program_type, attrs, feedback, learning_rate):
    """
    Update ProgramType CPD based on feedback.
    """
    cpt = cpt_counts["ProgramType"]
    parents = cpt["parents"]
    
    # Build parent state tuple from available attributes
    parent_state = build_parent_state(parents, attrs)
    
    if parent_state not in cpt["counts"]:
        #print(f"Parent state {parent_state} not found in ProgramType CPD")
        return
    
    if feedback == "accepted":
        # Reinforce the recommended type
        cpt["counts"][parent_state][program_type] += learning_rate
        #print(f"Reinforced {program_type} for context {parent_state}")
    
    elif feedback == "rejected":
        # Penalize the recommended type (but don't go below a minimum)
        current = cpt["counts"][parent_state][program_type]
        penalty = min(learning_rate * 0.5, current * 0.2)  # Max 20% reduction
        cpt["counts"][parent_state][program_type] = max(1, current - penalty)
        
        # Slightly boost alternatives
        all_types = cpt["state_names"]["ProgramType"]
        alternatives = [t for t in all_types if t != program_type]
        boost_per_alt = penalty / len(alternatives) if alternatives else 0
        
        for alt in alternatives:
            cpt["counts"][parent_state][alt] += boost_per_alt
        
        #print(f"Penalized {program_type}, boosted alternatives for context {parent_state}")
    
    # Rebuild and replace CPD
    new_cpd = build_cpd_from_counts("ProgramType", cpt)
    model.remove_cpds("ProgramType")
    model.add_cpds(new_cpd)


def update_program_genre_cpd(model, cpt_counts, program_type, program_genre, attrs, feedback, learning_rate):
    """
    Update ProgramGenre CPD based on feedback.
    ProgramGenre typically depends on ProgramType and possibly other context.
    """
    cpt = cpt_counts["ProgramGenre"]
    parents = cpt["parents"]
    
    # Build parent state - must include ProgramType
    attrs_with_type = dict(attrs)
    attrs_with_type["ProgramType"] = program_type
    
    parent_state = build_parent_state(parents, attrs_with_type)
    
    if parent_state not in cpt["counts"]:
        #print(f"Parent state {parent_state} not found in ProgramGenre CPD")
        return
    
    if feedback == "accepted":
        # Reinforce the recommended genre
        cpt["counts"][parent_state][program_genre] += learning_rate
        #print(f"Reinforced {program_genre} for Type={program_type}, context={parent_state}")
    
    elif feedback == "rejected":
        # Penalize the recommended genre
        current = cpt["counts"][parent_state][program_genre]
        penalty = min(learning_rate * 0.5, current * 0.2)
        cpt["counts"][parent_state][program_genre] = max(1, current - penalty)
        
        # Boost alternatives
        all_genres = cpt["state_names"]["ProgramGenre"]
        alternatives = [g for g in all_genres if g != program_genre]
        boost_per_alt = penalty / len(alternatives) if alternatives else 0
        
        for alt in alternatives:
            cpt["counts"][parent_state][alt] += boost_per_alt
        
        #print(f"Penalized {program_genre}, boosted alternatives for Type={program_type}")
    
    # Rebuild and replace CPD
    new_cpd = build_cpd_from_counts("ProgramGenre", cpt)
    model.remove_cpds("ProgramGenre")
    model.add_cpds(new_cpd)


def build_parent_state(parents, attrs):
    """
    Build a parent state tuple from available attributes.
    Uses None for missing parent values.
    """
    parent_values = []
    for p in parents:
        value = attrs.get(p)
        parent_values.append(value if value not in (None, "", "null") else None)
    
    return tuple(parent_values)


# ============================================================================
# Utility: Save/Load CPT counts for persistence
# ============================================================================

def save_cpt_counts(cpt_counts, filepath):
    """
    Save CPT counts to a JSON file for persistence across sessions.
    """
    import json
    
    # Convert defaultdict to regular dict for JSON serialization
    serializable = {}
    for var, info in cpt_counts.items():
        counts_dict = {
            str(k): dict(v) for k, v in info["counts"].items()
        }
        serializable[var] = {
            "parents": info["parents"],
            "state_names": info["state_names"],
            "counts": counts_dict
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"CPT counts saved to {filepath}")


def load_cpt_counts(filepath):
    """
    Load CPT counts from a JSON file.
    """
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    cpt_counts = {}
    for var, info in data.items():
        counts = defaultdict(lambda: defaultdict(float))
        for k_str, v_dict in info["counts"].items():
            k_tuple = eval(k_str) 
            counts[k_tuple] = defaultdict(float, v_dict)
        
        cpt_counts[var] = {
            "parents": info["parents"],
            "state_names": info["state_names"],
            "counts": counts
        }
    
    return cpt_counts
