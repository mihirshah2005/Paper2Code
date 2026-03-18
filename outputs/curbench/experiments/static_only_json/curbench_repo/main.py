"""
main.py

This is the main entry point for reproducing the experiments described in the CurBench paper.
It orchestrates the entire pipeline by loading the configuration, setting reproducibility seeds,
loading and preprocessing data, building backbone models, setting up a curriculum learning scheduler,
training the model, and evaluating its performance. The experiments are repeated over multiple seeds,
and the aggregated results (average and standard deviation) are reported.

The pipeline follows these steps:
  1. Load the configuration from "config.yaml" using the Config class.
  2. For each reproducibility seed, set the random seed for consistency.
  3. Load and preprocess the dataset (e.g., "cifar10") using DatasetLoader.
  4. Wrap dataset splits into DataLoaders.
  5. Build the backbone model with ModelManager (e.g., using "LeNet" for CV).
  6. Instantiate a curriculum scheduler (SelfPacedLearningScheduler in this example).
  7. Train the model using Trainer, which integrates the curriculum scheduler into the training loop.
  8. Evaluate the trained model using Evaluation, computing domain-specific metrics.
  9. Aggregate and report the results over multiple seeds.
  
All key configuration parameters (training settings, curriculum hyperparameters, etc.) are read from config.yaml.
"""

import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset_loader import DatasetLoader
from model import ModelManager
from curriculum import SelfPacedLearningScheduler
from trainer import Trainer
from evaluation import Evaluation

# Set up basic logging configuration.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch (including CUDA if available)
    to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d.", seed)


def aggregate_metric(metric_key: str, results_list: list) -> tuple:
    """
    Compute the average and standard deviation for a given metric key
    over a list of result dictionaries.
    
    Args:
        metric_key (str): The key for the metric to aggregate.
        results_list (list): List of result dictionaries.
    
    Returns:
        tuple: A tuple (average, standard deviation) of the metric values.
    """
    values = [res.get(metric_key, 0.0) for res in results_list]
    avg = np.mean(values)
    std = np.std(values)
    return avg, std


def main() -> None:
    """
    Main function orchestrates the entire experimental pipeline.
    """
    # 1. Load Configuration and Initialize Logging.
    config_obj = Config("config.yaml")
    config_dict = config_obj.params
    logger.info("Configuration loaded successfully:\n%s", config_dict)
    
    # 2. Retrieve Reproducibility Seeds.
    reproducibility_config = config_obj.get_reproducibility_config()
    seeds = reproducibility_config.get("seeds", [42])
    logger.info("Reproducibility seeds: %s", seeds)
    
    # Lists to collect training summaries and evaluation metrics for each seed.
    aggregated_train_summaries = []
    aggregated_eval_metrics = []
    
    # Default dataset and model selections.
    # Using "cifar10" as default dataset identifier; modify as needed.
    dataset_identifier: str = "cifar10"
    # For CV experiments, choose "LeNet" as default backbone model.
    default_model_name: str = "LeNet"
    
    # 3. Loop Over Reproducibility Seeds.
    for seed in seeds:
        logger.info("Starting experiment with seed: %d", seed)
        set_seed(seed)
        
        # 4. Data Loading and Preprocessing.
        dataset_loader = DatasetLoader(config_dict)
        raw_data = dataset_loader.load_data(dataset_identifier)
        logger.info("Dataset '%s' loaded successfully.", dataset_identifier)
        
        # Wrap dataset splits into DataLoaders if not already wrapped.
        # For CV datasets, batch_size is obtained from the training configuration.
        cv_training_config = config_obj.get_training_config("cv")
        batch_size: int = int(cv_training_config.get("batch_size", 50))
        
        if not isinstance(raw_data.get("train"), DataLoader):
            raw_data["train"] = DataLoader(raw_data["train"], batch_size=batch_size, shuffle=True)
        if not isinstance(raw_data.get("val"), DataLoader):
            raw_data["val"] = DataLoader(raw_data["val"], batch_size=batch_size, shuffle=False)
        if not isinstance(raw_data.get("test"), DataLoader):
            raw_data["test"] = DataLoader(raw_data["test"], batch_size=batch_size, shuffle=False)
        logger.info("Data splits wrapped into DataLoaders (batch_size=%d).", batch_size)
        
        # 5. Model Building.
        # Get the number of classes from the loaded data.
        num_classes: int = int(raw_data.get("num_classes", 10))
        logger.info("Number of classes in dataset: %d", num_classes)
        model_manager = ModelManager(config_dict)
        # For CV, pass additional kwargs (e.g., input_channels and image_size for CIFAR-10).
        model = model_manager.build_model(
            domain="cv",
            model_name=default_model_name,
            num_classes=num_classes,
            input_channels=3,
            image_size=32
        )
        logger.info("Model built using %s.", model.__class__.__name__)
        
        # 6. Curriculum Learning Scheduler Setup.
        curriculum_scheduler = SelfPacedLearningScheduler(config_dict)
        logger.info("Curriculum scheduler (SelfPacedLearningScheduler) initialized.")
        
        # 7. Training Process.
        trainer = Trainer(model=model, data=raw_data, curriculum=curriculum_scheduler, config=config_dict)
        train_summary = trainer.train()
        logger.info("Training completed for seed %d. Summary: %s", seed, train_summary)
        aggregated_train_summaries.append(train_summary)
        
        # 8. Evaluation.
        evaluator = Evaluation(model=model, data=raw_data, config=config_dict)
        eval_metrics = evaluator.evaluate()
        logger.info("Evaluation metrics for seed %d: %s", seed, eval_metrics)
        aggregated_eval_metrics.append(eval_metrics)
    
    # 9. Aggregate and Report Results.
    # Training metrics: total training time, final average loss, and max GPU memory.
    total_time_avg, total_time_std = aggregate_metric("total_time", aggregated_train_summaries)
    final_loss_avg, final_loss_std = aggregate_metric("final_avg_loss", aggregated_train_summaries)
    gpu_memory_avg, gpu_memory_std = aggregate_metric("max_gpu_memory_GB", aggregated_train_summaries)
    
    # Evaluation metrics: For CV, use accuracy.
    accuracy_avg, accuracy_std = aggregate_metric("accuracy", aggregated_eval_metrics)
    
    logger.info(
        "Aggregated Training Metrics: Total Time = %.4f sec (± %.4f), Final Avg Loss = %.4f (± %.4f), Max GPU Memory = %.4f GB (± %.4f)",
        total_time_avg, total_time_std, final_loss_avg, final_loss_std, gpu_memory_avg, gpu_memory_std
    )
    logger.info("Aggregated Evaluation Metrics: Accuracy = %.4f (± %.4f)", accuracy_avg, accuracy_std)
    
    # Final printed summary.
    print("Final Aggregated Results:")
    print("Training - Total Time: {:.4f} sec (± {:.4f}), Final Avg Loss: {:.4f} (± {:.4f}), Max GPU Memory: {:.4f} GB (± {:.4f})".format(
        total_time_avg, total_time_std, final_loss_avg, final_loss_std, gpu_memory_avg, gpu_memory_std))
    print("Evaluation - Accuracy: {:.4f} (± {:.4f})".format(accuracy_avg, accuracy_std))


if __name__ == "__main__":
    main()
