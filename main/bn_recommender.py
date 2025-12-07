from inference import load_model, query_model

model = load_model("model.pkl")

def recomendar_generos_bn(evidencia):
    # Realizamos la inferencia en la BN
    res = query_model(model, variables=["GeneroPrograma"], evidence=evidencia)

    # res será UN SOLO DiscreteFactor → no un dict
    dist = res

    # Obtener los valores posibles del nodo
    valores = dist.state_names["GeneroPrograma"]

    # Obtener probabilidades
    probs = dist.values

    # Asociar cada valor con su probabilidad
    recomendaciones = list(zip(valores, probs))

    # Ordenar de mayor a menor probabilidad
    recomendaciones.sort(key=lambda x: x[1], reverse=True)

    return recomendaciones
