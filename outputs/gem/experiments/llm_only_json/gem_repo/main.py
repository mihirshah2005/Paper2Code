"""
main.py

Main entry point for reproducing Graph Distillation with Eigenbasis Matching (GDEM).

This file performs the following steps:
  1. Loads configuration from "config.yaml".
  2. Instantiates a DatasetLoader to load the chosen dataset, compute the normalized Laplacian,
     and perform eigen-decomposition to extract the real graph’s spectral information.
  3. Computes the synthetic node count (N′) using the compression_ratio from the config.
  4. Instantiates the Model with random synthetic node features and a synthetic eigenbasis,
     initialized via a random orthogonal scheme (as a reproducible baseline).
  5. Sets up the Trainer (which enforces an alternating update schedule based on tau1 and tau2)
     and runs the training loop for the configured number of epochs.
     NOTE: A tau1 value of 0 implies that the synthetic eigenbasis is not updated; a warning is logged.
  6. After training, calls Model.reconstruct_graph() to generate the synthetic adjacency and Laplacian.
  7. Instantiates Evaluation to train a set of candidate GNNs (e.g., a 2-layer GCN, SGC)
     on the synthetic graph and evaluate their performance on the real graph’s test set.
  8. Repeats the entire process for num_runs experiments (default 10) for averaging and variance analysis,
     and then outputs a final aggregated summary.
  
All default values are set explicitly, and configuration parameters are read directly from config.yaml.
"""

import os
import logging
import yaml
import numpy as np
import torch

# Import project modules.
from dataset_loader import DatasetLoader, GraphData
from model import Model
from trainer import Trainer
from evaluation import Evaluation

def setup_logging() -> None:
    """
    Set up the root logger with DEBUG level and a standard formatter.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def main() -> None:
    # Step 1: Load configuration from "config.yaml"
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        logging.error("Configuration file 'config.yaml' not found.")
        exit(1)
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    logging.info("Configuration loaded from %s.", config_path)

    # Step 2: Load the dataset using DatasetLoader.
    dataset_loader = DatasetLoader(config)
    graph_data: GraphData = dataset_loader.load_data()
    num_real_nodes: int = graph_data.adjacency.shape[0]
    logging.info("Graph data loaded. Real graph has %d nodes.", num_real_nodes)

    # Step 3: Compute synthetic node count.
    synth_graph_cfg = config.get("synth_graph", {})
    raw_compression_ratio: float = synth_graph_cfg.get("compression_ratio", 0.15)
    # raw_compression_ratio is assumed already normalized to a fraction.
    num_synth_nodes: int = max(1, int(round(raw_compression_ratio * num_real_nodes)))
    logging.info("Synthetic node count (N'): %d (using compression_ratio = %.4f).", num_synth_nodes, raw_compression_ratio)

    # Step 4: Prepare Model parameters.
    model_params = {
        "num_synth_nodes": num_synth_nodes,
        "feature_dim": graph_data.node_features.shape[1],
        "loss_weights": config.get("training", {}).get("loss_weights", {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        "learning_rate_eigenvecs": config.get("training", {}).get("learning_rate_eigenvecs", 0.01),
        "learning_rate_feat": config.get("training", {}).get("learning_rate_feat", 1e-05),
        "r_k": synth_graph_cfg.get("r_k", 0.9)
    }
    logging.info("Model parameters: %s", str(model_params))
    # Note: Synthetic eigenbasis (U'_K) is initialized using a random orthogonal method.
    # Alternative initialization (e.g., via SBM) is not implemented in this baseline.

    # Step 5: Define number of experimental runs.
    training_cfg = config.get("training", {})
    num_runs: int = int(training_cfg.get("num_runs", 10))
    logging.info("Number of experimental runs: %d", num_runs)

    evaluations_across_runs = []  # To store evaluation metrics for each run.

    # Step 6: Loop over experimental runs.
    for run in range(num_runs):
        logging.info("Starting experimental run %d/%d.", run + 1, num_runs)
        # Instantiate a new Model for the current run.
        model_instance = Model(model_params)

        # Instantiate Trainer with the current model, graph data, and training configuration.
        trainer = Trainer(model_instance, graph_data, config)
        # Check tau1 value; if tau1 <= 0, warn the user.
        tau1: int = int(training_cfg.get("tau1", 0))
        if tau1 <= 0:
            logging.warning("tau1 is set to %d. This implies that the synthetic eigenbasis (U'_K) will not be updated during training.", tau1)

        # Run the training process.
        A_prime, L_prime = trainer.train()
        logging.info("Run %d: Training complete. Synthetic graph reconstructed.", run + 1)

        # Step 7: Evaluate the distilled synthetic graph using candidate GNNs.
        eval_config = config.get("evaluation", {})
        evaluator = Evaluation(model_instance, graph_data, eval_config)
        run_eval_results = evaluator.evaluate()
        logging.info("Run %d: Evaluation results: %s", run + 1, str(run_eval_results))
        evaluations_across_runs.append(run_eval_results)

    # Step 8: Aggregate evaluation metrics across all runs.
    aggregated_results = {}
    candidate_names = set()
    for run_results in evaluations_across_runs:
        for key in run_results:
            if key != "diagnostics":
                candidate_names.add(key)
    candidate_names = list(candidate_names)

    # Initialize aggregated lists for each candidate.
    for cand in candidate_names:
        aggregated_results[cand] = []

    for run_results in evaluations_across_runs:
        for cand in candidate_names:
            if cand in run_results:
                acc = run_results[cand].get("accuracy", 0)
                aggregated_results[cand].append(acc)

    final_summary = {}
    for cand in candidate_names:
        acc_list = aggregated_results[cand]
        mean_acc = float(np.mean(acc_list))
        std_acc = float(np.std(acc_list))
        final_summary[cand] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}

    # Aggregate diagnostic information (e.g., synthetic total variation).
    tv_list = [run_results.get("diagnostics", {}).get("synthetic_total_variation", None) 
               for run_results in evaluations_across_runs if run_results.get("diagnostics", {}).get("synthetic_total_variation", None) is not None]
    avg_tv = float(np.mean(tv_list)) if tv_list else None
    final_summary["diagnostics"] = {"average_synthetic_total_variation": avg_tv}

    # Log and print the final aggregated evaluation summary.
    logging.info("Final aggregated evaluation results over %d runs: %s", num_runs, str(final_summary))
    print("\nFinal Evaluation Summary:")
    for key, metrics in final_summary.items():
        print(f"{key}: {metrics}")


if __name__ == "__main__":
    setup_logging()
    main()
