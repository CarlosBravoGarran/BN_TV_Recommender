
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


def recommend_gender(evidence, model):
    infer = VariableElimination(model)
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