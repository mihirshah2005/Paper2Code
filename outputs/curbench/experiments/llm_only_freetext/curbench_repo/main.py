"""main.py

Main entry point for running the CurBench benchmark experiments.

This script performs the following steps:
    1. Loads the configuration from config.yaml using the Config class.
    2. Iterates over multiple random seeds for reproducibility.
    3. For each seed:
         a. Sets the random seeds for Python, NumPy, and PyTorch.
         b. Loads and preprocesses the dataset using DatasetLoader.
         c. Instantiates the appropriate backbone model via ModelManager based on the domain.
         d. Initializes the curriculum learning scheduler (SelfPacedLearningScheduler).
         e. Creates a Trainer to run the training loop with curriculum scheduling.
         f. Evaluates the trained model using the Evaluation class.
    4. Aggregates evaluation results (performance and complexity metrics) across seeds.
    5. Logs and prints the final aggregated outcomes.
    
All modules and hyperparameters rely on configuration values specified in config.yaml.
"""

import os
import sys
import time
import logging
import random
import numpy as np
import torch

# Import project modules
from config import Config
from dataset_loader import DatasetLoader
from model import ModelManager
from curriculum import SelfPacedLearningScheduler
from trainer import Trainer
from evaluation import Evaluation

# Configure module-level logger for main
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_global_seeds(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}.")


def aggregate_results(results: list) -> dict:
    """
    Aggregates a list of result dictionaries (one per seed) by computing the mean and standard deviation
    for each performance and complexity metric.
    
    Args:
        results (list): List of dictionaries containing evaluation results.
                        Each dictionary has keys 'performance' and 'complexity'.
                        
    Returns:
        dict: Aggregated results in the form:
              {
                  'performance': {metric1: {'mean': x, 'std': y}, ...},
                  'complexity': {metric2: {'mean': x, 'std': y}, ...}
              }
    """
    aggregated = {"performance": {}, "complexity": {}}
    # Initialize containers for performance and complexity metrics
    performance_metrics = {}
    complexity_metrics = {}
    
    # Collect metrics from each result
    for result in results:
        perf = result.get("performance", {})
        comp = result.get("complexity", {})
        for key, value in perf.items():
            performance_metrics.setdefault(key, []).append(value)
        for key, value in comp.items():
            complexity_metrics.setdefault(key, []).append(value)
    
    # Compute mean and standard deviation for performance metrics
    for key, values in performance_metrics.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        aggregated["performance"][key] = {"mean": mean_val, "std": std_val}
    
    # Compute mean and standard deviation for complexity metrics
    for key, values in complexity_metrics.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        aggregated["complexity"][key] = {"mean": mean_val, "std": std_val}
    
    return aggregated


def main() -> None:
    """
    Executes the complete experimental pipeline:
       - Loads configuration.
       - Iterates over multiple seeds.
       - For each seed, loads data, builds model, trains with curriculum scheduling,
         and evaluates performance and complexity.
       - Aggregates and logs results across seeds.
    """
    # Step 1. Load configuration from config.yaml
    try:
        config_loader = Config("config.yaml")
        config_dict = config_loader.load()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    logger.info("Configuration loaded successfully.")
    
    # Get reproducibility seeds from configuration
    seeds = config_dict.get("reproducibility", {}).get("seeds", [42])
    if not isinstance(seeds, list) or len(seeds) == 0:
        seeds = [42]
    logger.info(f"Using seeds for reproducibility: {seeds}")

    # List to store evaluation summaries for each seed run
    evaluation_results = []

    # Default dataset identifier. Here we choose "cifar10" as default.
    default_dataset_id: str = "cifar10"
    
    # For each seed, run experiment independently
    for seed in seeds:
        logger.info(f"Starting experiment with seed {seed}...")
        # Step 2. Set global seeds for reproducibility
        set_global_seeds(seed)
        
        # Step 3. Data Loading and Preprocessing
        # Instantiate DatasetLoader with configuration and dataset identifier.
        try:
            dataset_loader = DatasetLoader(config=config_dict, dataset_id=default_dataset_id)
            data_dict = dataset_loader.load_data()
            logger.info("Data loaded and preprocessed successfully.")
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            continue  # Skip to next seed if data loading fails
        
        # Step 4. Model Instantiation using ModelManager
        try:
            model_manager = ModelManager(config=config_dict)
            # For CV domain "cifar10", specify default dataset_info
            dataset_info = {
                "num_classes": 10,
                "in_channels": 3,
                "image_size": 32,
                # Additional parameters can be added as needed
            }
            # Choose model from config training->cv->models, default to first model e.g., "LeNet"
            cv_models = config_dict.get("training", {}).get("cv", {}).get("models", ["LeNet"])
            chosen_model_name = cv_models[0] if isinstance(cv_models, list) and len(cv_models) > 0 else "LeNet"
            model = model_manager.build_model(domain="cv", model_name=chosen_model_name, dataset_info=dataset_info)
            logger.info(f"Model '{chosen_model_name}' instantiated successfully.")
        except Exception as e:
            logger.error(f"Model instantiation failed: {e}")
            continue
        
        # Step 5. Curriculum Learning Scheduler Initialization
        try:
            curriculum_scheduler = SelfPacedLearningScheduler(config=config_dict)
            logger.info("Curriculum scheduler initialized successfully.")
        except Exception as e:
            logger.error(f"Curriculum scheduler initialization failed: {e}")
            continue
        
        # Step 6. Training Loop Orchestration using Trainer
        try:
            trainer = Trainer(model=model, data=data_dict, curriculum_scheduler=curriculum_scheduler, config=config_dict)
            trained_model, training_history = trainer.train()
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            continue
        
        # Step 7. Evaluation and Metrics Calculation
        try:
            evaluator = Evaluation(model=trained_model, data=data_dict, config=config_dict)
            eval_summary = evaluator.evaluate()
            logger.info(f"Evaluation results for seed {seed}: {eval_summary}")
            evaluation_results.append(eval_summary)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            continue

    # Step 8. Aggregate results across all seeds
    if evaluation_results:
        aggregated = aggregate_results(evaluation_results)
        logger.info("Aggregated Evaluation Results Across Seeds:")
        logger.info("Performance Metrics:")
        for metric, values in aggregated["performance"].items():
            logger.info(f"  {metric}: Mean = {values['mean']:.4f}, Std = {values['std']:.4f}")
        logger.info("Complexity Metrics:")
        for metric, values in aggregated["complexity"].items():
            logger.info(f"  {metric}: Mean = {values['mean']:.4f}, Std = {values['std']:.4f}")
        
        # Print final summary in a structured format
        print("\nFinal Aggregated Evaluation Results:")
        print("Performance Metrics:")
        for metric, values in aggregated["performance"].items():
            print(f"  {metric}: Mean = {values['mean']:.4f}, Std = {values['std']:.4f}")
        print("Complexity Metrics:")
        for metric, values in aggregated["complexity"].items():
            print(f"  {metric}: Mean = {values['mean']:.4f}, Std = {values['std']:.4f}")
    else:
        logger.error("No valid evaluation results were obtained from any seed run.")


if __name__ == "__main__":
    main()
