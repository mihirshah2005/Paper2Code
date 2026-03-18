"""main.py

This is the entry point for the FiT reproduction pipeline.
It orchestrates the following:
  - Loading of configuration from config.yaml.
  - Setting seeds for reproducibility.
  - Initializing the DatasetLoader for training and evaluation data.
  - Constructing the FiT Model, loading the pretrained VAE, and ensuring that
    patchification and positional embedding interpolation are properly set.
  - Creating the Trainer instance to run the training loop.
  - Creating the Evaluation instance to perform diffusion sampling and compute evaluation metrics.
  - Logging and printing comprehensive results.

All default values and configuration parameters are set based on config.yaml.
"""

import os
import random
import yaml
import numpy as np
import torch

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """
    Main class to orchestrate the experiment.
    
    Attributes:
        config (dict): Experiment configuration loaded from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the Main instance with the provided configuration.
        
        Args:
            config (dict): Configuration dictionary.
        """
        self.config: dict = config

    def run_experiment(self) -> None:
        """
        Main method to run the complete experiment:
          1. Set random seeds.
          2. Prepare data loaders for training and evaluation.
          3. Initialize the FiT model and load the pretrained VAE.
          4. Create and run the training loop via the Trainer.
          5. Run the evaluation procedure via the Evaluation class.
        """
        # Set random seeds for reproducibility.
        seed: int = self.config.get("seed", 42)
        self.set_seed(seed)
        print(f"Using random seed: {seed}")

        # Create the DatasetLoader and obtain DataLoaders.
        dataset_loader_instance: DatasetLoader = DatasetLoader(self.config)
        train_dataloader = dataset_loader_instance.load_data()
        # For simplicity, using the same loader for evaluation.
        eval_dataloader = dataset_loader_instance.load_data()
        print("Data loaders initialized.")

        # Instantiate the Model.
        model_instance: Model = Model(self.config)
        # Determine model variant (e.g., "B/2" or "XL/2"); default to "B/2" if not specified.
        variant: str = self.config.get("model", {}).get("variant", "B/2")
        print(f"Initializing model with variant: {variant}")
        # Load the pretrained VAE using the provided path; default if not provided.
        pretrained_vae_path: str = self.config.get("model", {}).get("pretrained_vae",
                                                                     "huggingface/stabilityai/sd-vae-ft-ema")
        model_instance.load_pretrained_vae(pretrained_vae_path)
        print(f"Pretrained VAE loaded from: {pretrained_vae_path}")

        # Set up dummy base positional embeddings if not already defined.
        transformer_cfg: dict = self.config.get("model", {}).get("transformer", {})
        hidden_size: int = transformer_cfg.get("hidden_size", 768)
        max_token_length: int = self.config.get("model", {}).get("max_token_length", 256)
        if not hasattr(model_instance, "base_positional_embeddings"):
            model_instance.base_positional_embeddings = torch.randn(max_token_length, hidden_size)
            print("Assigned dummy base positional embeddings to the model.")
        # If interpolate_positional_embeddings method is not implemented, assign a dummy function.
        if not hasattr(model_instance, "interpolate_positional_embeddings"):
            def dummy_interpolate_positional_embeddings(
                original_pe: torch.Tensor,
                target_shape: tuple,
                method: str,
                scale_factors: tuple
            ) -> torch.Tensor:
                """
                Dummy positional embedding interpolation that repeats original embeddings to fill the target grid.
                
                Args:
                    original_pe (Tensor): Original positional embeddings of shape [L_max, embedding_dim].
                    target_shape (tuple): Target grid dimensions (H_latent, W_latent).
                    method (str): Chosen interpolation method.
                    scale_factors (tuple): Scale factors for (height, width).
                
                Returns:
                    Tensor: Interpolated positional embeddings of shape [1, H_latent * W_latent, embedding_dim].
                """
                H_latent, W_latent = target_shape
                token_count = H_latent * W_latent
                embedding_dim = original_pe.shape[1]
                interpolated = torch.zeros(1, token_count, embedding_dim, device=original_pe.device)
                for i in range(token_count):
                    idx = i % original_pe.shape[0]
                    interpolated[0, i] = original_pe[idx]
                return interpolated

            model_instance.interpolate_positional_embeddings = dummy_interpolate_positional_embeddings
            print("Assigned dummy interpolate_positional_embeddings method to the model.")

        # Instantiate the Trainer and run training if configured.
        trainer_instance: Trainer = Trainer(model_instance, train_dataloader, self.config)
        run_training: bool = self.config.get("run_training", True)
        if run_training:
            print("Starting training ...")
            trainer_instance.train()
        else:
            print("Skipping training as per configuration setting.")

        # After training, initialize Evaluation and run evaluation.
        print("Starting evaluation ...")
        evaluation_instance: Evaluation = Evaluation(model_instance, eval_dataloader, self.config)
        results: dict = evaluation_instance.evaluate()

        # Log and print evaluation metrics.
        metrics: dict = results.get("metrics", {})
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        gen_images = results.get("generated_images", None)
        if gen_images is not None:
            print("Generated images tensor shape:", gen_images.shape)
        else:
            print("No generated images obtained from evaluation.")

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set random seed for Python, numpy, and torch for reproducibility.
        
        Args:
            seed (int): Random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file. If the file does not exist,
    use default configuration values.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: The loaded configuration dictionary.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config: dict = yaml.safe_load(config_file)
        print(f"Configuration loaded from {config_path}")
    else:
        print(f"{config_path} not found. Using default configuration values.")
        config = {
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 256,
                "training_steps_fit_b2": 400000,
                "training_steps_fit_xl2": 1800000,
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "ema_decay": 0.9999,
                "checkpoint_interval": 1000
            },
            "model": {
                "patch_size": 2,
                "max_token_length": 256,
                "variant": "B/2",
                "transformer": {
                    "hidden_size": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "ffn_hidden_size": 3072,
                    "time_embed_dim": 768,
                    "positional_embedding": "2D RoPE",
                    "attention": "Masked MHSA",
                    "ffn": "SwiGLU"
                },
                "pretrained_vae": "huggingface/stabilityai/sd-vae-ft-ema"
            },
            "diffusion": {
                "num_sampling_steps": 250,
                "noise_schedule": "DDPM"
            },
            "data": {
                "dataset_path": "./data/imagenet",
                "resize_max_area": 65536,
                "augmentation": "Horizontal Flip"
            },
            "extrapolation": {
                "methods": ["PI", "EI", "NTK", "YaRN", "VisionNTK", "VisionYaRN"],
                "scale_factor": "max(max(H_test, W_test)/L_train, 1.0)"
            },
            "evaluation": {
                "fid_steps": 250,
                "metrics": ["FID", "sFID", "IS", "Precision", "Recall"],
                "num_samples": 10,
                "resolution": [256, 256],
                "H_train": 16,
                "W_train": 16
            },
            "run_training": True,
            "seed": 42
        }
    return config


def main() -> None:
    """
    Main function to load configuration, instantiate the experiment, and run it.
    """
    config: dict = load_config("config.yaml")
    main_experiment: Main = Main(config)
    main_experiment.run_experiment()


if __name__ == "__main__":
    main()
