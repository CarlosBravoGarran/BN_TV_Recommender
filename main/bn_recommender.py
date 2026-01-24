
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

CORE_NODES = [
    "UserAge",
    "UserGender",
    "HouseholdType",
    "TimeOfDay",
    "DayType",
    "ProgramType",
    "ProgramGenre",
    "ProgramDuration",
]

def _core_model(full_model: DiscreteBayesianNetwork) -> DiscreteBayesianNetwork:
    # Subgrafo inducido por CORE_NODES
    core_edges = [e for e in full_model.edges() if e[0] in CORE_NODES and e[1] in CORE_NODES]
    core = DiscreteBayesianNetwork(core_edges)

    # Copiar CPDs del modelo completo (solo las del core)
    core_cpds = [cpd for cpd in full_model.get_cpds() if cpd.variable in CORE_NODES]
    core.add_cpds(*core_cpds)

    # IMPORTANTE: que el core sea válido
    core.check_model()
    return core


def recommend_gender(evidence, model):
    core = _core_model(model)
    infer = VariableElimination(core)
    res = infer.query(
        variables=["ProgramGenre"],
        evidence=evidence,
        show_progress=False
    )

    dist = res
    values = dist.state_names["ProgramGenre"]
    probs = dist.values

    recommendations = list(zip(values, probs))
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations