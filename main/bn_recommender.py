
from pgmpy.inference import VariableElimination

def recomendar_generos_bn(evidencia, model):
    infer = VariableElimination(model)
    res = infer.query(
        variables=["GeneroPrograma"],
        evidence=evidencia,
        show_progress=False
    )

    dist = res
    valores = dist.state_names["GeneroPrograma"]
    probs = dist.values

    recomendaciones = list(zip(valores, probs))
    recomendaciones.sort(key=lambda x: x[1], reverse=True)

    return recomendaciones
