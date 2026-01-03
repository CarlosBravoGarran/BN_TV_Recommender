"""graph_builder.py

Utilities to build a Bayesian Network (structure + CPDs) from a CSV and
save/visualize the result.
This module groups the creation/fitting logic from the original `graph_gen.py`
so it can be reused from other scripts (e.g., inference).
"""

import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BDeu
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import pickle


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV and return a DataFrame."""
    return pd.read_csv(path)


def learn_structure(df: pd.DataFrame):
    """Learn the structure using Hill Climb and BDeu.
    Returns the model object (pgmpy networkx.DiGraph-like) with the edges.
    """
    hill_climb = HillClimbSearch(df)
    best_structure = hill_climb.estimate(scoring_method=BDeu(df, equivalent_sample_size=100))
    return best_structure


def save_edges(best_structure, path: str = "best_model_edges.csv") -> None:
    """Save the best model edges to a CSV (source,target)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("source,target\n")
        for source, target in best_structure.edges():
            f.write(f"{source},{target}\n")


def build_and_fit_model(
    csv_path: str = "tv_bn_dataset.csv",
    save_edges_path: str = "best_model_edges.csv",
    save_model_path: str = "model.pkl",
    prior_type: str = "BDeu",
    equivalent_sample_size: int = 100,
    visualize: bool = True,
) -> tuple[DiscreteBayesianNetwork, pd.DataFrame]:
    """Full pipeline: load data, learn structure, build model, fit CPDs.
    Optionally saves edges and the model (pickle) and visualizes the network.
    Returns (model, data_frame).
    """
    data_frame = load_data(csv_path)
    best_structure = learn_structure(data_frame)

    if save_edges_path:
        save_edges(best_structure, save_edges_path)

    model = DiscreteBayesianNetwork(best_structure.edges())
    model.fit(
        data_frame,
        estimator=BayesianEstimator,
        prior_type=prior_type,
        equivalent_sample_size=equivalent_sample_size,
    )

    if save_model_path:
        save_model(model, save_model_path)

    if visualize:
        visualize_model(model)

    return model, data_frame


def visualize_model(model: DiscreteBayesianNetwork, figsize=(10, 8)) -> None:
    """Draw the network using networkx/matplotlib.
    It does not display the figure if running without a display; caller decides.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        arrowsize=20,
        edgecolors="black",
    )
    plt.title("Automatically learned Bayesian Network", fontsize=14)
    try:
        plt.show()
    except Exception:
        pass


def save_model(model: DiscreteBayesianNetwork, path: str) -> None:
    """Save the model to a file using pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> DiscreteBayesianNetwork:
    """Load a model saved with `save_model`."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_cpds_to_text(model, path: str = "model_cpds.txt") -> None:
    """Save all CPDs (conditional probability tables) to a single text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Conditional probability tables (CPDs)\n")
        f.write("=" * 80 + "\n\n")

        for cpd in model.get_cpds():
            f.write(f"CPD for {cpd.variable}:\n")
            f.write(str(cpd))
            f.write("\n" + "-" * 80 + "\n")

    print(f"CPDs saved to '{path}'")


if __name__ == "__main__":
    model, data_frame = build_and_fit_model(csv_path="main/tv_bn_dataset.csv")
    save_cpds_to_text(model, path="model_cpds.txt")
    
