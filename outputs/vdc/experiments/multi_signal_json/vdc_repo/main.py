"""
main.py

This is the main entry point for the VDC (Versatile Data Cleanser)
experiment and training pipeline. The code performs the following steps:

1. Loads configuration parameters from "config.yaml".
2. Loads datasets (CIFAR-10, ImageNet-100, and ImageNet-Dog) using the DatasetLoader,
   and injects dirty samples (poisoned samples and noisy labels) into the training data.
3. Instantiates the detection pipeline modules:
   - Visual Question Generation (VQG)
   - Visual Question Answering (VQA)
   - Visual Answer Evaluation (VAE)
   These are integrated into VDCDetector which processes each training sample.
4. Purifies the training dataset by filtering out samples that are detected as "dirty".
5. Trains a ResNet-18 classifier on the purified training dataset using the Trainer module.
6. Constructs a combined evaluation test set (with both clean and triggered samples)
   and evaluates the classifier performance (ACC) and attack success rate (ASR)
   (and, if applicable, detection metrics TPR and FPR) using the Evaluation module.
7. Logs and prints final metrics.

All configuration parameters (e.g., detection threshold, number of questions, hyperparameters)
are read from config.yaml. Default values are provided for any missing parameters.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

# Import modules from project files.
from dataset_loader import DatasetLoader, ListDataset
from utils import load_config, set_random_seed, inject_badnets_trigger
from vqg import VQG
from vqa import VQA
from vae import VAE
from detection import VDCDetector
from trainer import Trainer
from evaluation import Evaluation

# Set up a global logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main() -> None:
    """Main entry point for the experiment."""
    # Step 1: Load configuration from config.yaml.
    config_path: str = os.path.join(os.getcwd(), "config.yaml")
    try:
        config: Dict[str, Any] = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}. Exiting.")
        sys.exit(1)

    # Set random seed for reproducibility.
    set_random_seed(42)

    # Step 2: Dataset Loading and Dirty Sample Injection.
    # Choose dataset; default to "CIFAR-10". (Could be parameterized.)
    selected_dataset: str = "CIFAR-10"
    logger.info(f"Selected dataset: {selected_dataset}")

    # Instantiate the DatasetLoader.
    # For hybrid dirty sample creation, hybrid_dirty flag is set as False by default.
    dataset_loader = DatasetLoader(config=config, hybrid_dirty=False)
    # Load original datasets.
    data: Dict[str, Dict[str, Any]] = dataset_loader.load_data()
    if selected_dataset not in data:
        logger.error(f"Dataset '{selected_dataset}' not found in loaded data. Exiting.")
        sys.exit(1)

    original_train_dataset = data[selected_dataset]["train"]
    original_test_dataset = data[selected_dataset]["test"]
    logger.info(f"Loaded training and test splits for {selected_dataset}.")

    # Inject dirty samples into the training data.
    data_injected: Dict[str, Dict[str, Any]] = dataset_loader.inject_dirty_samples(data)
    injected_train_dataset = data_injected[selected_dataset]["train"]
    logger.info(f"Dirty sample injection complete for training data of '{selected_dataset}'.")

    # Step 3: Module Instantiation for Detection Pipeline.
    # Get detection settings from configuration.
    detection_config: Dict[str, Any] = config.get("detection", {})
    detection_threshold: float = detection_config.get("threshold", 0.5)
    general_q_count: int = detection_config.get("general_questions_count", 2)
    label_spec_q_dict: Dict[str, int] = detection_config.get("label_specific_questions", {})
    label_spec_q_count: int = label_spec_q_dict.get(selected_dataset, 4)

    logger.info(f"Detection parameters: threshold={detection_threshold}, "
                f"general_questions_count={general_q_count}, "
                f"label_specific_questions for {selected_dataset}={label_spec_q_count}.")

    # Instantiate VQG with default general templates and configured question counts.
    vqg_instance = VQG(general_questions_count=general_q_count, label_specific_count=label_spec_q_count)
    
    # Build VQA configuration.
    vqa_config: Dict[str, Any] = {
        "model_name": config.get("api", {}).get("instruct_blip", {}).get("model", "Instruct-BLIP"),
        "input_size": (224, 224),
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    vqa_instance = VQA(model_config=vqa_config)

    # Create VAE instance; use evaluation configuration from config if available.
    eval_config: Dict[str, Any] = config.get("evaluation", {})
    vae_instance = VAE(eval_config=eval_config)

    # Instantiate the VDCDetector.
    detector_instance = VDCDetector(vqg=vqg_instance, vqa=vqa_instance, vae=vae_instance, threshold=detection_threshold)
    logger.info("Detection pipeline modules instantiated.")

    # Step 4: Detection and Purification of Training Data.
    # Iterate over each sample in the injected training dataset and use detector to decide if clean.
    purified_samples: List[Tuple[Any, int]] = []
    total_samples: int = 0
    detected_clean: int = 0

    # Assume injected_train_dataset is an instance of ListDataset containing (image, label) tuples.
    for sample in injected_train_dataset:
        total_samples += 1
        image, label = sample
        # Call detector.detect_sample. Convert label to string.
        try:
            is_clean: bool = detector_instance.detect_sample(image, str(label))
        except Exception as e:
            logger.error(f"Detection error for sample with label {label}: {e}")
            is_clean = False
        if is_clean:
            purified_samples.append((image, label))
            detected_clean += 1

    logger.info(f"Detection completed: {detected_clean} out of {total_samples} training samples are considered clean.")

    # Build purified training dataset.
    purified_train_dataset = ListDataset(purified_samples)

    # Step 5: Training the Classifier.
    # Instantiate a ResNet-18 classifier and adjust final layer based on number of classes.
    if selected_dataset == "CIFAR-10":
        num_classes: int = 10
    elif selected_dataset == "ImageNet-100":
        num_classes = 100
    elif selected_dataset == "ImageNet-Dog":
        # Try to get number of classes from dataset attributes; if not available, default to 10.
        try:
            num_classes = len(injected_train_dataset.dataset.classes)
        except Exception:
            num_classes = 10
    else:
        num_classes = 10

    model_classifier: nn.Module = models.resnet18(pretrained=False)
    model_classifier.fc = nn.Linear(model_classifier.fc.in_features, num_classes)
    logger.info(f"Classifier model instantiated: ResNet-18 with {num_classes} classes.")

    # Instantiate the Trainer.
    trainer_instance = Trainer(
        model=model_classifier,
        train_dataset=purified_train_dataset,
        dataset_name=selected_dataset,
        config=config
    )
    logger.info("Starting classifier training on purified dataset...")
    trainer_instance.train()
    logger.info("Training completed.")

    # Step 6: Evaluation (ACC and ASR Computation).
    # Prepare a combined test dataset including both clean and triggered samples.
    clean_test_samples: List[Tuple[Any, int, bool]] = []
    triggered_test_samples: List[Tuple[Any, int, bool]] = []

    # For every sample in the original test dataset:
    for sample in original_test_dataset:
        # The sample from torchvision.datasets returns (image, label); treat as clean.
        try:
            image, label = sample
        except Exception as e:
            logger.error(f"Error unpacking test sample: {e}")
            continue
        clean_test_samples.append((image, label, False))
        # For triggered sample, apply the same poisoning trigger as used in training.
        try:
            triggered_image = inject_badnets_trigger(image, dataset=selected_dataset, config=config)
        except Exception as e:
            logger.error(f"Error applying trigger to test sample: {e}")
            triggered_image = image
        triggered_test_samples.append((triggered_image, label, True))

    # Combine the two sets.
    combined_test_samples: List[Tuple[Any, int, bool]] = clean_test_samples + triggered_test_samples
    evaluation_dataset = ListDataset(combined_test_samples)
    logger.info(f"Combined evaluation dataset constructed with {len(clean_test_samples)} clean samples and "
                f"{len(triggered_test_samples)} triggered samples.")

    # Instantiate Evaluation.
    evaluation_instance = Evaluation(
        model=model_classifier,
        test_data=evaluation_dataset,
        detector=detector_instance,
        config=config
    )
    logger.info("Starting evaluation...")
    final_metrics: Dict[str, float] = evaluation_instance.evaluate()

    # Step 7: Final Logging of Metrics.
    logger.info("Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")

    print("Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
