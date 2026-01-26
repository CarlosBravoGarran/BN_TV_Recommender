"""
graph_builder.py

Utilities to build a Bayesian Network (structure + CPDs) from a CSV dataset
(ConsumersProfile) using Hill Climbing and BDeu.

This module:
- Learns a population-level BN structure (profile + context → content)
- Applies semantic constraints (domain knowledge)
- Fits CPDs using Bayesian estimation
- Saves edges, CPDs and the full model
- Visualizes the learned graph

"""

import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BDeu, ExpertKnowledge


# ============================================================================
# CONFIG
# ============================================================================

BASE_COLUMNS = [
    "UserAge",
    "UserGender",
    "HouseholdType",
    "TimeOfDay",
    "DayType",
    "ProgramType",
    "ProgramGenre",
    "ProgramDuration",
]

PROFILE_CONTEXT = [
    "UserAge",
    "UserGender",
    "HouseholdType",
    "TimeOfDay",
    "DayType",
]

CONTENT = [
    "ProgramType",
    "ProgramGenre",
    "ProgramDuration",
]


WHITELIST = [
    ("UserAge", "ProgramType"),
    ("UserGender", "ProgramType"),
    ("HouseholdType", "ProgramType"),
    ("TimeOfDay", "ProgramType"),
    ("DayType", "ProgramType"),
    ("ProgramType", "ProgramGenre"),
    ("ProgramType", "ProgramDuration"),
]

BLACKLIST = []
for target in PROFILE_CONTEXT:
    for source in BASE_COLUMNS:
        if source != target:
            BLACKLIST.append((source, target))

for node in BASE_COLUMNS:
    BLACKLIST.append(("ProgramDuration", node))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ============================================================================
# STRUCTURE LEARNING
# ============================================================================

def learn_structure(df: pd.DataFrame, equivalent_sample_size: int = 100):
    df_bn = df[BASE_COLUMNS].copy()

    hc = HillClimbSearch(df_bn)

    ek = ExpertKnowledge(
        required_edges=WHITELIST,
        forbidden_edges=BLACKLIST,
    )

    structure = hc.estimate(
        scoring_method=BDeu(df_bn, equivalent_sample_size=equivalent_sample_size),
        expert_knowledge=ek,
    )

    edges = list(structure.edges())
    return edges, df_bn


# ============================================================================
# MODEL BUILDING & FITTING
# ============================================================================

def build_and_fit_model(
    csv_path: str = "main/consumers_profile.csv",
    save_edges_path: str = "main/outputs/model_edges.csv",
    save_model_path: str = "main/outputs/model.pkl",
    save_cpds_path: str = "main/outputs/model_cpds.txt",
    prior_type: str = "BDeu",
    equivalent_sample_size: int = 100,
    visualize: bool = False,
):
    df = load_data(csv_path)

    edges, df_bn = learn_structure(df, equivalent_sample_size=equivalent_sample_size)

    model = DiscreteBayesianNetwork(edges)

    model.fit(
        df_bn,
        estimator=BayesianEstimator,
        prior_type=prior_type,
        equivalent_sample_size=equivalent_sample_size,
    )

    model.check_model()

    if save_edges_path:
        save_edges(edges, save_edges_path)

    if save_model_path:
        save_model(model, save_model_path)

    if save_cpds_path:
        save_cpds_to_text(model, save_cpds_path)

    if visualize:
        visualize_model(model)

    return model, df_bn, edges


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_model(model: DiscreteBayesianNetwork, figsize=(12, 9)) -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, seed=42)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=1400,
        node_color="lightblue",
        font_size=9,
        arrowsize=20,
        edgecolors="black",
    )

    plt.title("Learned Bayesian Network (Base: Profile/Context → Content)")
    plt.tight_layout()
    plt.show()


# ============================================================================
# SAVE / LOAD UTILITIES
# ============================================================================

def save_edges(edges, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("source,target\n")
        for s, t in edges:
            f.write(f"{s},{t}\n")


def save_model(model: DiscreteBayesianNetwork, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> DiscreteBayesianNetwork:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cpds_to_text(model: DiscreteBayesianNetwork, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Conditional Probability Tables (CPDs)\n")
        f.write("=" * 80 + "\n\n")
        for cpd in model.get_cpds():
            f.write(f"CPD for {cpd.variable}:\n")
            f.write(str(cpd))
            f.write("\n" + "-" * 80 + "\n")
    print(f"CPDs saved to '{path}'")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    build_and_fit_model(
        csv_path="main/consumers_profile.csv",
        visualize=True,
    )
