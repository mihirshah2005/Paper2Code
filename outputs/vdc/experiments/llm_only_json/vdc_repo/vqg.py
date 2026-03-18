"""
vqg.py

This module implements the VQG (Visual Question Generation) class for the Versatile Data Cleanser (VDC).
It generates two kinds of questions for a given image-label sample:
  - General questions: drawn from a fixed, configurable list.
  - Label-specific questions: generated dynamically by calling the ChatGPT API.
  
The module uses caching to avoid redundant API calls and ensures reproducibility through controlled randomness.
All configuration values are loaded from a YAML file (default "config.yaml").
It adheres strictly to configuration parameters such as the number of questions to generate, and
ensures that unknown dataset identifiers are rejected explicitly.

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

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

import openai

from utils import (
    load_config,
    generate_api_cache_key,
    get_cached_response,
    cache_api_response,
    init_seed,
)

# Setup logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ask_chatgpt(prompt: str, model: str = "gpt-3.5-turbo", max_retries: int = 2, delay: float = 2.0) -> str:
    """
    Calls the ChatGPT API with the provided prompt and returns the response text.
    Retries the API call on failure up to max_retries times with a delay between retries.

    Args:
        prompt (str): The prompt string to send to the ChatGPT API.
        model (str): The ChatGPT model to use (default "gpt-3.5-turbo").
        max_retries (int): Maximum number of retries on failure.
        delay (float): Delay in seconds between retries.

    Returns:
        str: The API response text.

    Raises:
        Exception: If all retries fail.
    """
    for attempt in range(max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            # Assume the response text is contained in the first choice.
            message = response.choices[0].message.get("content", "").strip()
            if message:
                return message
            else:
                raise ValueError("Empty response from ChatGPT API.")
        except Exception as exc:
            logger.error(f"ChatGPT API call failed on attempt {attempt+1}/{max_retries+1}: {exc}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise Exception("ChatGPT API call failed after all retries.") from exc


class VQG:
    """
    Visual Question Generation (VQG) module for the Versatile Data Cleanser (VDC).
    
    Generates two types of visual questions for a given label:
      - General questions (template-based)
      - Label-specific questions (generated via ChatGPT)
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        general_templates (List[str]): Predefined list of general question templates.
        prompt_version (str): Version identifier for the prompt template.
        general_count (int): Number of general questions to generate.
        ls_questions_mapping (Dict[str, int]): Mapping from dataset id to the expected
                                                number of label-specific questions.
        seed (int): Random seed for reproducibility.
        chatgpt_model (str): The ChatGPT model name.
    """

    def __init__(
        self,
        general_templates: Optional[List[str]] = None,
        prompt_version: str = "v1",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the VQG module with provided templates and configuration.
        If configuration is not provided, it loads from 'config.yaml'.

        Args:
            general_templates (Optional[List[str]]): List of general question templates.
                If None, a default list is used.
            prompt_version (str): Version identifier for the prompt.
            config (Optional[Dict[str, Any]]): Configuration dictionary.
        """
        self.config: Dict[str, Any] = config if config is not None else load_config("config.yaml")
        self.prompt_version: str = prompt_version

        # Set a reproducible seed
        self.seed: int = self.config.get("seed", 42)
        init_seed(self.seed)

        # Set general question templates.
        if general_templates is None:
            # Default general question templates, e.g., inspired by Table 10 in Appendix E.
            self.general_templates: List[str] = [
                "Describe the image in detail.",
                "Describe the image briefly.",
                "Summarize the content of the image in a few words.",
                "Provide a concise description of the image."
            ]
        else:
            self.general_templates = general_templates

        # Get the expected count for general questions from configuration.
        detection_config = self.config.get("detection", {})
        self.general_count: int = detection_config.get("general_questions_count", 2)

        # Get mapping for label-specific questions per dataset.
        self.ls_questions_mapping: Dict[str, int] = detection_config.get("label_specific_questions", {})
        if not self.ls_questions_mapping:
            logger.error("No label-specific questions mapping found in configuration under detection.label_specific_questions.")
            raise ValueError("Missing label-specific questions mapping in configuration.")

        # Set ChatGPT model from config.api.chatgpt, with default fallback
        api_config = self.config.get("api", {}).get("chatgpt", {})
        self.chatgpt_model: str = api_config.get("model", "gpt-3.5-turbo")

        logger.info(f"VQG initialized with seed {self.seed}, prompt version '{self.prompt_version}', "
                    f"{self.general_count} general questions and label-specific mapping: {self.ls_questions_mapping}.")

    def generate_general_questions(self, label: str) -> List[str]:
        """
        Generates a list of general visual questions using predefined templates.
        The method shuffles the templates using a seeded random instance for controlled randomness.
        
        Args:
            label (str): The label associated with the image (not used for general questions but kept for interface consistency).
        
        Returns:
            List[str]: A list of general question strings.
        """
        # Create a local copy of templates and shuffle using a reproducible random generator.
        templates_copy: List[str] = self.general_templates.copy()
        rnd: random.Random = random.Random(self.seed)
        rnd.shuffle(templates_copy)
        
        # Select the first N templates (where N = self.general_count).
        if len(templates_copy) < self.general_count:
            logger.warning(f"Requested {self.general_count} general questions, but only {len(templates_copy)} templates are available.")
            selected_templates: List[str] = templates_copy
        else:
            selected_templates = templates_copy[: self.general_count]

        logger.info(f"Generated general questions: {selected_templates}")
        return selected_templates

    def generate_label_specific_questions(self, label: str, dataset_id: str) -> List[str]:
        """
        Generates label-specific visual questions for a given label by interfacing with the ChatGPT API.
        Employs caching to avoid redundant API calls. The prompt instructs ChatGPT to return a JSON array
        of exactly N questions.

        Args:
            label (str): The label for which to generate questions.
            dataset_id (str): Identifier for the dataset (e.g., "CIFAR-10", "ImageNet-100", or "ImageNet-Dog").

        Returns:
            List[str]: A list of label-specific question strings.

        Raises:
            ValueError: If the dataset_id is not present in the configuration mapping.
            Exception: If API response parsing fails and no fallback can be provided.
        """
        if dataset_id not in self.ls_questions_mapping:
            error_msg = f"Unknown dataset identifier '{dataset_id}'; no mapping for label-specific question count found."
            logger.error(error_msg)
            raise ValueError(error_msg)
        expected_count: int = self.ls_questions_mapping[dataset_id]

        # Construct the prompt for ChatGPT API.
        prompt: str = (
            f"Please generate exactly {expected_count} insightful visual questions to verify if the label '{label}' "
            f"accurately describes the visual content of an image. "
            f"Return your answer as a JSON array of strings without any additional text."
        )
        # Build a caching key from dataset_id, label, expected_count and prompt_version.
        cache_params: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "label": label,
            "expected_count": expected_count,
            "prompt_version": self.prompt_version,
            "function": "generate_label_specific_questions"
        }
        cache_key: str = generate_api_cache_key(cache_params)
        cached_response: Optional[str] = get_cached_response(cache_key)
        if cached_response is not None:
            api_response: str = cached_response
            logger.info(f"Using cached ChatGPT response for key {cache_key}.")
        else:
            # Call ChatGPT API via helper function.
            try:
                api_response = ask_chatgpt(prompt, model=self.chatgpt_model)
                # Cache the API response.
                cache_api_response(cache_key, api_response)
                logger.info(f"API response cached with key {cache_key}.")
            except Exception as exc:
                logger.error(f"ChatGPT API call failed: {exc}")
                # Fallback: generate a default set of questions.
                default_questions = [f"Is the object in the image a {label}?" for _ in range(expected_count)]
                logger.warning("Returning fallback label-specific questions due to API failure.")
                return default_questions

        # Attempt to parse the response as a JSON array.
        try:
            questions = json.loads(api_response)
            if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
                raise ValueError("Parsed JSON is not a list of strings.")
            if len(questions) != expected_count:
                logger.warning(f"Expected {expected_count} questions, but got {len(questions)}. Attempting fallback parsing.")
                raise ValueError("Incorrect number of questions in JSON response.")
        except Exception as parse_exc:
            logger.error(f"Failed to parse ChatGPT response as JSON: {parse_exc}")
            # Fallback parsing: split by newlines and filter non-empty lines.
            fallback_questions = [line.strip() for line in api_response.splitlines() if line.strip()]
            # Ensure we return exactly expected_count questions.
            if len(fallback_questions) < expected_count:
                # Fallback default if insufficient questions.
                fallback_questions = [f"Is the object in the image a {label}?" for _ in range(expected_count)]
                logger.warning("Insufficient questions parsed; using default fallback questions.")
            else:
                fallback_questions = fallback_questions[:expected_count]
            questions = fallback_questions

        logger.info(f"Generated label-specific questions for label '{label}' on dataset '{dataset_id}': {questions}")
        return questions

    def generate_all_questions(self, label: str, dataset_id: str) -> Dict[str, List[str]]:
        """
        Generates both general and label-specific questions for the given label and dataset identifier.

        Args:
            label (str): The label for which to generate questions.
            dataset_id (str): The dataset identifier.

        Returns:
            Dict[str, List[str]]: A dictionary with keys 'general' and 'label_specific' mapping to lists of questions.
        """
        general: List[str] = self.generate_general_questions(label)
        label_specific: List[str] = self.generate_label_specific_questions(label, dataset_id)
        return {"general": general, "label_specific": label_specific}
