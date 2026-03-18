"""
vqa.py

This module implements the VQA (Visual Question Answering) class for the Versatile Data Cleanser (VDC).
It wraps a multimodal model (preferably Instruct‑BLIP) to generate textual answers for a provided image and question.
The module loads the model according to configuration settings in config.yaml, applies controlled decoding parameters,
and preprocesses the image properly. The generated answer is returned as a string to be passed downstream to the
VAE (Visual Answer Evaluation) module.

Dependencies:
    - numpy==1.21.0
    - torch==1.9.0
    - torchvision==0.10.0
    - transformers==4.15.0
    - openai==0.27.0
    - requests==2.26.0
    - PyYAML (for YAML configuration parsing)
    
Author: Your Name
Date: YYYY-MM-DD
"""

import logging
from typing import Any, Dict

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, AutoTokenizer, AutoModelForConditionalGeneration

from utils import load_config, init_seed

# Set up module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VQA:
    """
    VQA class implements the Visual Question Answering module of the VDC pipeline.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        device (torch.device): The device on which the model is allocated.
        model_name (str): The name of the multimodal model to use (from configuration).
        processor (BlipProcessor): Processor used for preprocessing image and text.
        model (BlipForQuestionAnswering): The loaded multimodal model.
        max_length (int): Maximum length for the generated answer.
        temperature (float): Decoding temperature.
        num_beams (int): Number of beams used in beam search.
        prompt_template (str): Template string for dynamic prompt formatting.
    """
    
    def __init__(self, model_config: Dict[str, Any]) -> None:
        """
        Initializes the VQA module.
        
        Loads the Instruct-BLIP model based on configuration settings.
        If a specialized Instruct-BLIP model is not available via configuration,
        falls back to using a generic HuggingFace model (Salesforce/blip-vqa-base).
        
        Decoding parameters (max_length, temperature, num_beams) are read from the configuration
        (or use default values if not specified). The image preprocessing pipeline is handled by the processor.
        
        Args:
            model_config (Dict[str, Any]): Configuration dictionary for the model. Expected to include
                                           API parameters under "api.instruct_blip" and decoding parameters
                                           under "decoding" (if available).
        """
        self.config: Dict[str, Any] = model_config
        init_seed(self.config.get("seed", 42))
        
        # Device allocation: Use GPU if available.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Retrieve the instruct_blip model configuration.
        api_config: Dict[str, Any] = self.config.get("api", {}).get("instruct_blip", {})
        configured_model_name: str = api_config.get("model", "Instruct-BLIP")
        # Fallback: if the configured model name is "Instruct-BLIP" or empty, substitute with a known vqa model.
        if not configured_model_name or configured_model_name == "Instruct-BLIP":
            # Log a warning and fallback.
            logger.warning("No explicit Instruct-BLIP model provided. Falling back to 'Salesforce/blip-vqa-base'.")
            self.model_name = "Salesforce/blip-vqa-base"
        else:
            self.model_name = configured_model_name
        
        # Load Decoding parameters from a potential "decoding" section in config.
        decoding_config: Dict[str, Any] = self.config.get("decoding", {})
        self.max_length: int = decoding_config.get("max_length", 50)
        self.temperature: float = decoding_config.get("temperature", 0.7)
        self.num_beams: int = decoding_config.get("num_beams", 1)
        
        # Dynamic prompt formatting - if provided in config use it; else default to simple pass-through.
        # The prompt template can include a placeholder {question} for dynamic formatting.
        self.prompt_template: str = self.config.get("vqa_prompt", "{question}")
        
        try:
            # Initialize the processor and model via HuggingFace Transformers.
            # Use the BlipProcessor and BlipForQuestionAnswering for visual Q&A.
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
            logger.info(f"Loaded VQA model '{self.model_name}' successfully.")
        except Exception as exc:
            logger.error(f"Failed to load the specified Instruct-BLIP model '{self.model_name}': {exc}")
            # Fallback strategy using generic Auto classes (though not optimal for multimodal input)
            try:
                from transformers import AutoTokenizer, AutoModelForConditionalGeneration
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForConditionalGeneration.from_pretrained(self.model_name)
                # Dummy processor using the tokenizer (note: this fallback may not
                # process images correctly, so it's advisable to provide a valid model)
                self.processor = type("DummyProcessor", (), {
                    "image_processor": lambda image: image,
                    "tokenizer": self.tokenizer,
                    "__call__": lambda self, image, text, return_tensors: self.tokenizer(text, return_tensors="pt")
                })()
                logger.warning("Using fallback generic model for VQA. This may not support multimodal inputs properly.")
            except Exception as fallback_exc:
                logger.error(f"Fallback model loading failed: {fallback_exc}")
                raise RuntimeError("VQA model initialization failed and no viable fallback found.") from fallback_exc
        
        # Set model to evaluation mode and move it to the configured device.
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"VQA model set to eval mode and moved to device {self.device}.")

    def answer_question(self, image: Any, question: str) -> str:
        """
        Generates an answer for the given image and question using the loaded VQA model.
        
        Steps:
          1. Preprocess the image and apply the dynamic prompt formatting on the provided question.
          2. Tokenize the formatted prompt and prepare the inputs (including the image) as required by the model.
          3. Invoke the model’s generate method with provided decoding parameters.
          4. Decode the generated token ids into a human readable answer string.
        
        Args:
            image (Any): The raw input image (expected as PIL Image).
            question (str): The question string to be answered by the model.
        
        Returns:
            str: The generated answer text.
        """
        try:
            # Dynamic prompt formatting: if a prompt template is provided, format the question.
            formatted_question: str = self.prompt_template.format(question=question)
            
            # Process the image and text using the processor.
            inputs = self.processor(image=image, text=formatted_question, return_tensors="pt")
            # Move inputs to device.
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(self.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            
            # Generate answer with specified decoding parameters.
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs.get("input_ids", None),
                    pixel_values=inputs.get("pixel_values", None),
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    temperature=self.temperature,
                    early_stopping=True
                )
            
            # Decode output token ids to text; use the processor's tokenizer.
            answer: str = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = answer.strip()
            if not answer:
                logger.error("Generated empty answer from VQA model.")
                return ""
            logger.info(f"VQA generated answer: {answer}")
            return answer
        except Exception as exc:
            logger.error(f"VQA answer generation failed: {exc}")
            return ""
