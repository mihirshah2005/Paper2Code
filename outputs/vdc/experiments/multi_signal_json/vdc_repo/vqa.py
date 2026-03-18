"""
vqa.py

This module implements the VQA (Visual Question Answering) class for the VDC project.
It wraps around a multimodal model (e.g., Instruct-BLIP) to answer visual questions based 
on an input image and a text question. The module validates the provided model configuration,
sets up a robust image preprocessing pipeline using torchvision.transforms, loads the model 
onto the desired device, and provides the method answer_question to obtain a textual answer.

The expected model_config schema is:
    {
        "model_name": str,            # e.g., "Instruct-BLIP" (default: "Instruct-BLIP")
        "input_size": tuple or int,     # e.g., (224, 224) if int then assumed square (default: (224, 224))
        "normalization": {
            "mean": [float, float, float],  # e.g., [0.485, 0.456, 0.406]
            "std":  [float, float, float]    # e.g., [0.229, 0.224, 0.225]
        },
        "device": str                 # "cuda" or "cpu"; if absent, automatically determined
    }

The answer_question method preprocesses the image, then calls the multimodal model’s inference 
method. It checks for "generate", "inference", or "forward" methods to ensure compatibility.

Default values are provided for missing configuration parameters.
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

# Set up module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# DummyInstructBLIP simulates a multimodal model (e.g., Instruct-BLIP).
class DummyInstructBLIP:
    """
    A dummy implementation of the Instruct-BLIP model interface for VQA.
    This dummy model supports 'generate', 'inference', and 'forward' methods.
    It simply returns a canned answer for demonstration purposes.
    """
    def __init__(self, model_name: str = "Instruct-BLIP") -> None:
        self.model_name = model_name
        self.device = "cpu"

    def to(self, device: str) -> "DummyInstructBLIP":
        self.device = device
        return self

    def generate(self, image_tensor: torch.Tensor, question: str) -> str:
        # For simulation purposes, return a dummy answer.
        return f"Dummy answer for question '{question}' using generate method."

    def inference(self, image_tensor: torch.Tensor, question: str) -> str:
        return f"Dummy answer for question '{question}' using inference method."

    def forward(self, image_tensor: torch.Tensor, question: str) -> str:
        return f"Dummy answer for question '{question}' using forward method."


class VQA:
    """
    VQA class for answering visual questions using a multimodal model (e.g., Instruct-BLIP).
    
    Public Methods:
        __init__(model_config: dict) -> None
        answer_question(image: Any, question: str) -> str
    """
    def __init__(self, model_config: Dict[str, Any]) -> None:
        """
        Initializes the VQA module.
        
        Args:
            model_config (Dict[str, Any]): Configuration dictionary for the model.
                Expected keys:
                    - "model_name": (str) Name of the model (default: "Instruct-BLIP").
                    - "input_size": (tuple or int) Expected image size (default: (224, 224)).
                    - "normalization": (dict) with keys "mean" [0.485, 0.456, 0.406]
                                  and "std" [0.229, 0.224, 0.225] (default values used if missing).
                    - "device": (str) "cuda" or "cpu" (if missing, automatically set).
        """
        # Validate and set model name.
        self.model_name: str = model_config.get("model_name", "Instruct-BLIP")
        if not isinstance(self.model_name, str):
            logger.warning("model_name should be a string. Using default 'Instruct-BLIP'.")
            self.model_name = "Instruct-BLIP"
        
        # Validate and set input_size.
        input_size_val = model_config.get("input_size", (224, 224))
        if isinstance(input_size_val, int):
            self.input_size: Tuple[int, int] = (input_size_val, input_size_val)
        elif isinstance(input_size_val, (list, tuple)) and len(input_size_val) == 2:
            self.input_size = tuple(input_size_val)
        else:
            logger.warning("input_size not provided or invalid; defaulting to (224, 224).")
            self.input_size = (224, 224)
        
        # Validate normalization parameters.
        normalization = model_config.get("normalization", {})
        self.mean = normalization.get("mean", [0.485, 0.456, 0.406])
        self.std = normalization.get("std", [0.229, 0.224, 0.225])
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and len(self.mean) == 3 and len(self.std) == 3):
            logger.warning("Normalization parameters invalid; using default mean and std.")
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        
        # Determine and set device.
        self.device: str = model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"VQA module using device: {self.device}")
        
        # Build image preprocessing pipeline.
        self.preprocess = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Load the multimodal model.
        try:
            # In a real-world scenario, replace DummyInstructBLIP with actual model loading code.
            self.model = DummyInstructBLIP(model_name=self.model_name)
            self.model.to(self.device)
            logger.info(f"Loaded model '{self.model_name}' on device {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise e

    def answer_question(self, image: Any, question: str) -> str:
        """
        Performs inference for a given image and question, returning the model's answer as a string.
        
        Args:
            image (Any): The input image. Expected to be a PIL Image or an array convertible to PIL.
            question (str): The text question to ask about the image.
        
        Returns:
            str: The textual answer generated by the multimodal model.
        
        Raises:
            Exception: If image conversion, preprocessing, or inference fails.
        """
        # Convert to PIL Image if needed.
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as e:
                logger.error(f"Error converting input to PIL Image: {e}")
                raise e
        
        # Preprocess the image.
        try:
            image_tensor = self.preprocess(image)  # Tensor with shape (C, H, W)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension.
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            raise e
        
        # Prepare and perform model inference.
        try:
            if hasattr(self.model, "generate"):
                answer = self.model.generate(image_tensor, question)
            elif hasattr(self.model, "inference"):
                answer = self.model.inference(image_tensor, question)
            elif hasattr(self.model, "forward"):
                answer = self.model.forward(image_tensor, question)
            else:
                error_msg = ("The loaded model does not support a recognized inference method "
                             "('generate', 'inference', or 'forward').")
                logger.error(error_msg)
                raise AttributeError(error_msg)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise e

        # Postprocess the answer: ensure it is a clean string.
        try:
            if not isinstance(answer, str):
                answer = str(answer)
            answer = answer.strip()
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            raise e
        
        return answer


if __name__ == "__main__":
    # Quick test for the VQA module.
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Define a sample model configuration (normally loaded from config.yaml)
    sample_model_config = {
        "model_name": "Instruct-BLIP",
        "input_size": (224, 224),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Instantiate the VQA module.
    vqa_instance = VQA(model_config=sample_model_config)
    
    # Create a dummy sample image (a blank 256x256 image).
    sample_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    sample_question = "What is the main object in the image?"
    
    # Use the answer_question method.
    answer = vqa_instance.answer_question(sample_image, sample_question)
    print(f"Answer: {answer}")
