from collections import defaultdict
import itertools
import numpy as np
from pgmpy.factors.discrete import TabularCPD

from graph_builder import load_model


# ============================================================
# Inicialización de conteos a partir de CPDs existentes
# ============================================================

def initialize_cpt_counts(model, virtual_sample_size=100):
    """
    Convierte las CPDs del modelo en conteos iniciales en memoria.
    """
    cpt_counts = {}

    for cpd in model.get_cpds():
        variable = cpd.variable
        parents = cpd.variables[1:]
        state_names = cpd.state_names

        counts = defaultdict(lambda: defaultdict(float))
        values = cpd.values

        for idx, state in enumerate(state_names[variable]):
            for parent_idx in np.ndindex(values.shape[1:]):
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


# ============================================================
# Reconstrucción segura de CPDs desde conteos
# ============================================================

def build_cpd_from_counts(variable, cpt_info):
    parents = cpt_info["parents"]
    state_names = cpt_info["state_names"]
    counts = cpt_info["counts"]

    var_states = state_names[variable]

    # SOLO combinaciones válidas de estados de padres
    parent_states = [state_names[p] for p in parents]
    valid_parent_combinations = list(itertools.product(*parent_states))

    values = []

    for v in var_states:
        row = []
        for parent_state in valid_parent_combinations:
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


# ============================================================
# Aplicación de feedback del usuario
# ============================================================

def apply_feedback(model, cpt_counts, state):
    """
    Aplica feedback de un estado y actualiza CPTs afectadas.
    """

    feedback = state.get("user_feedback")
    if feedback is None:
        return

    # 1. Mapear feedback a variable latente
    satisfaccion = "alta" if feedback == "accepted" else "baja"

    attrs = state["atributos_bn"]
    genero_programa = state["last_recommendation"]

    # --------------------------------------------------------
    # CPT: Recomendado | Satisfaccion, PopularidadPrograma
    # --------------------------------------------------------

    cpt = cpt_counts["Recomendado"]
    parent_state = (
        satisfaccion,
        attrs.get("PopularidadPrograma", "media")
    )

    # IGNORAR feedback incompatible con la BN
    if parent_state in cpt["counts"]:
        recomendado = "sí" if feedback == "accepted" else "no"
        cpt["counts"][parent_state][recomendado] += 1

        new_cpd = build_cpd_from_counts("Recomendado", cpt)
        model.remove_cpds("Recomendado")
        model.add_cpds(new_cpd)

    # --------------------------------------------------------
    # CPT: GeneroPrograma | Satisfaccion, DuracionPrograma,
    #                      TipoEmision, InteresPrevio
    # --------------------------------------------------------

    cpt = cpt_counts["GeneroPrograma"]

    # Valor neutro fijo de InteresPrevio (no se aprende por feedback)
    interes_previo_fijo = cpt["state_names"]["InteresPrevio"][0]

    parent_state = (
        satisfaccion,
        attrs["DuracionPrograma"],
        attrs.get("TipoEmision", "desconocido"),
        interes_previo_fijo,
    )

    if parent_state in cpt["counts"]:
        cpt["counts"][parent_state][genero_programa] += 1

        new_cpd = build_cpd_from_counts("GeneroPrograma", cpt)
        model.remove_cpds("GeneroPrograma")
        model.add_cpds(new_cpd)

    model.check_model()


# ============================================================
# Ejecución
# ============================================================

if __name__ == "__main__":
    model = load_model("main/outputs/model.pkl")
    cpt_counts = initialize_cpt_counts(model)

    import json
    with open("main/states.json", "r", encoding="utf-8") as f:
        states = json.load(f)

    for state in states:
        apply_feedback(model, cpt_counts, state)
