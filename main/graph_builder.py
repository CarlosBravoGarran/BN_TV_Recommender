"""
graph_builder.py

Utilities to build a Bayesian Network (structure + CPDs) from a CSV dataset
(ConsumersProfile) using Hill Climbing and BDeu.

This module:
- Learns a population-level BN structure
- Applies semantic constraints (domain knowledge)
- Fits CPDs using Bayesian estimation
- Saves edges, CPDs and the full model
- Visualizes the learned graph

This BN represents user profile + context → content attributes.
"""

import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BDeu
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from pgmpy.estimators import ExpertKnowledge


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


# ============================================================================
# STRUCTURE LEARNING
# ============================================================================

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

for target in PROFILE_CONTEXT:
    for source in PROFILE_CONTEXT + CONTENT:
        if source != target:
            BLACKLIST.append((source, target))

for node in PROFILE_CONTEXT + CONTENT:
    BLACKLIST.append(("ProgramDuration", node))


def learn_structure_restricted(df: pd.DataFrame):
    hill_climb = HillClimbSearch(df)

    expert_knowledge = ExpertKnowledge(
        required_edges=WHITELIST,
        forbidden_edges=BLACKLIST,
    )

    structure = hill_climb.estimate(
        scoring_method=BDeu(df, equivalent_sample_size=100),
        expert_knowledge=expert_knowledge,
    )

    return list(structure.edges())

def learn_structure(df: pd.DataFrame):
    BN_COLUMNS = [
        'UserAge',
        'UserGender',
        'HouseholdType',
        'TimeOfDay',
        'DayType',
        'ProgramType',
        'ProgramGenre',
        'ProgramDuration',
    ]

    df_bn = df[BN_COLUMNS]

    hill_climb = HillClimbSearch(df_bn)
    learned_structure = hill_climb.estimate(
        scoring_method=BDeu(df_bn, equivalent_sample_size=100)
    )

    edges = list(learned_structure.edges())

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
    """
    Full pipeline:
    - Load dataset
    - Learn BN structure
    - Build DiscreteBayesianNetwork
    - Fit CPDs using Bayesian estimation
    - Save edges, model and CPDs
    - Optionally visualize the graph
    """

    df = load_data(csv_path)

    BN_COLUMNS = [
    'UserAge',
    'UserGender',
    'HouseholdType',
    'TimeOfDay',
    'DayType',
    'ProgramType',
    'ProgramGenre',
    'ProgramDuration',
]

    df_bn = df[BN_COLUMNS]
    edges = learn_structure_restricted(df_bn)


    if save_edges_path:
        save_edges(edges, save_edges_path)

    model = DiscreteBayesianNetwork(edges)

    FEEDBACK_NODES = [
        # History
        "SeenBefore",
        "PreviousContentType",
        "PreviousGenre",

        # Feedback
        "RecommendationAccepted",
        "UserSatisfaction",
        "ExplicitFeedback",
        "WatchRatio",
        "Abandoned",
        "RepeatContent",
    ]

    model.add_nodes_from(FEEDBACK_NODES)

    # Fit CPDs
    model.fit(
        df_bn,
        estimator=BayesianEstimator,
        prior_type=prior_type,
        equivalent_sample_size=equivalent_sample_size,
    )

    # Validate only nodes that have data / CPDs
    for node in df_bn.columns:
        assert model.get_cpds(node) is not None, f"Missing CPD for {node}"


    # Save model
    if save_model_path:
        save_model(model, save_model_path)

    # Save CPDs
    if save_cpds_path:
        save_cpds_to_text(model, save_cpds_path)

    # Visualize
    if visualize:
        visualize_model(model)

    return model, df_bn


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_model(model: DiscreteBayesianNetwork, figsize=(12, 9)) -> None:
    """Visualize the Bayesian Network using networkx and matplotlib."""

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
        font_size=10,
        arrowsize=20,
        edgecolors="black",
    )

    plt.title("Learned Bayesian Network (Profile → Content)", fontsize=14)
    plt.tight_layout()

    try:
        plt.show()
    except Exception:
        pass


# ============================================================================
# SAVE / LOAD UTILITIES
# ============================================================================

def save_edges(edges, path: str) -> None:
    """Save learned edges to CSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("source,target\n")
        for source, target in edges:
            f.write(f"{source},{target}\n")


def save_model(model: DiscreteBayesianNetwork, path: str) -> None:
    """Save the BN model using pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> DiscreteBayesianNetwork:
    """Load a previously saved BN model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cpds_to_text(model: DiscreteBayesianNetwork, path: str) -> None:
    """Save all CPDs to a readable text file."""
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
    model, df_bn = build_and_fit_model(
        csv_path="main/consumers_profile.csv",
        visualize=True,
    )
