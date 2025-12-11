"""inference.py

Funciones para realizar inferencias sobre un modelo de pgmpy (VariableElimination).
Este módulo usa el modelo (objeto DiscreteBayesianNetwork) generado por
`graph_builder.py`.
"""

import pickle
from pgmpy.inference import VariableElimination
from graph_builder import build_and_fit_model


def load_model(path: str):
    """Carga un modelo guardado con pickle."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def query_model(model, variables, evidence= None):
    """Ejecuta una consulta de probabilidad marginal condicionada sobre el modelo.
    variables: lista de nombres de variables a consultar.
    evidence: diccionario variable->valor.
    Devuelve el resultado de VariableElimination.query (objeto de pgmpy).
    """
    infer = VariableElimination(model)
    result = infer.query(variables=variables, evidence=evidence or {})
    return result


if __name__ == "__main__":

    model_path = "main/outputs/model.pkl"
    model = load_model(model_path)

    evidence = {"EdadUsuario": "mayor", 
                "Hora": "noche", 
                "DiaSemana": "fin_semana", 
                "DuracionPrograma": "media", 
                "PopularidadPrograma": "alta", 
                "InteresPrevio": "entretenimiento"}
    res = query_model(model, variables=["GeneroPrograma"], evidence=evidence)
    
    print("\nGénero recomendado para usuario:")
    print(res)
