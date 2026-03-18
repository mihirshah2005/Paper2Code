"""
detection.py

This module implements the VDCDetector class for the Versatile Data Cleanser (VDC).
The VDCDetector integrates the following modules:
  - VQG: Visual Question Generation (must provide "generate_general_questions(label)" and
          "generate_label_specific_questions(label, dataset_id)" methods).
  - VQA: Visual Question Answering (must provide "answer_question(image, question)").
  - VAE: Visual Answer Evaluation (must provide "evaluate_answer(answer, expected, question_type)").

For each sample (image, label), the detector:
  1. Generates two sets of questions (general and label-specific) with a retry mechanism if
     the count does not match the expected configuration.
  2. Uses the VQA module (with caching and exponential backoff) to obtain answers for each question.
  3. Evaluates each answer using the VAE module; for both general and label-specific questions,
     the expected answer is assumed to be "yes".
  4. Computes a vote-based matching score (sᵢ) and marks the sample as dirty if score < threshold.
  
The detect_sample() method returns a dictionary with:
    - "is_dirty": bool (True if sample is detected as dirty)
    - "score": float (matching score)
    - "details": List[Dict[str, Any]] (diagnostic info for each question evaluation)

Configuration parameters (from config.yaml) that are used include:
  detection:
    threshold: (e.g., 0.5)
    general_questions_count: (e.g., 2)
    label_specific_questions: a mapping of dataset id to count (e.g., {"CIFAR-10": 4, ...})
    max_retries: (optional, default 3)
    vqa_max_retries: (optional, default 3)
    vqa_initial_delay: (optional, default 1.0)

Usage Example:
    from detection import VDCDetector
    # vqg, vqa, vae are already instantiated module objects.
    detector = VDCDetector(vqg, vqa, vae, threshold=0.5, dataset_id="CIFAR-10", config=config)
    result = detector.detect_sample(image, "cat")
    if result["is_dirty"]:
        # mark sample as dirty (i.e., remove from dataset)
"""

import logging
import time
import random
from typing import Any, Dict, List, Optional

from utils import (
    load_config,
    generate_api_cache_key,
    get_cached_response,
    cache_api_response,
    init_seed,
)

# Set up module-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VDCDetector:
    def __init__(self, vqg: Any, vqa: Any, vae: Any, threshold: float, dataset_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the VDCDetector instance.
        
        Args:
            vqg: An instance of the VQG module.
            vqa: An instance of the VQA module.
            vae: An instance of the VAE module.
            threshold: Detection threshold (e.g., 0.5); samples with matching score below this are marked dirty.
            dataset_id: Identifier of the dataset (e.g., "CIFAR-10", "ImageNet-100", "ImageNet-Dog").
            config: Optional configuration dictionary. If not provided, loaded from "config.yaml".
        
        Raises:
            ValueError: For invalid configuration values.
            AttributeError: If required methods are missing in the injected modules.
        """
        if config is None:
            config = load_config("config.yaml")
        self.config: Dict[str, Any] = config
        self.dataset_id: str = dataset_id

        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("Invalid threshold; must be a positive number.")
        self.threshold: float = float(threshold)

        # Validate that vqg has required methods.
        if not (hasattr(vqg, "generate_general_questions") and callable(getattr(vqg, "generate_general_questions"))):
            raise AttributeError("VQG instance must have a callable method 'generate_general_questions'.")
        if not (hasattr(vqg, "generate_label_specific_questions") and callable(getattr(vqg, "generate_label_specific_questions"))):
            raise AttributeError("VQG instance must have a callable method 'generate_label_specific_questions'.")
        if not (hasattr(vqa, "answer_question") and callable(getattr(vqa, "answer_question"))):
            raise AttributeError("VQA instance must have a callable method 'answer_question'.")
        if not (hasattr(vae, "evaluate_answer") and callable(getattr(vae, "evaluate_answer"))):
            raise AttributeError("VAE instance must have a callable method 'evaluate_answer'.")
        self.vqg = vqg
        self.vqa = vqa
        self.vae = vae

        # Load detection configuration.
        detection_conf: Dict[str, Any] = self.config.get("detection", {})
        self.general_q_count: int = detection_conf.get("general_questions_count", 2)
        if not isinstance(self.general_q_count, int) or self.general_q_count <= 0:
            raise ValueError("Invalid configuration: 'detection.general_questions_count' must be a positive integer.")

        ls_q_mapping: Dict[str, Any] = detection_conf.get("label_specific_questions", {})
        if dataset_id not in ls_q_mapping:
            raise ValueError(f"Label-specific questions count not configured for dataset '{dataset_id}'.")
        self.label_specific_q_count: int = ls_q_mapping[dataset_id]
        if not isinstance(self.label_specific_q_count, int) or self.label_specific_q_count <= 0:
            raise ValueError(f"Invalid configuration: 'label_specific_questions' count for {dataset_id} must be a positive integer.")

        self.max_retries: int = detection_conf.get("max_retries", 3)
        # Reading VQA-related retry settings from detection config.
        self.vqa_max_retries: int = detection_conf.get("vqa_max_retries", 3)
        self.vqa_initial_delay: float = detection_conf.get("vqa_initial_delay", 1.0)
        # Expected answer for evaluation. For both general and label-specific questions, the correct answer is assumed "yes".
        self.general_expected: str = detection_conf.get("general_expected", "yes")
        
        # Ensure reproducibility.
        init_seed(self.config.get("seed", 42))

        logger.info(f"VDCDetector initialized for dataset '{dataset_id}' with threshold {self.threshold}, "
                    f"{self.general_q_count} general and {self.label_specific_q_count} label-specific questions, "
                    f"max_retries={self.max_retries}, VQA max retries={self.vqa_max_retries}.")

    def _get_vqa_answer(self, image: Any, question: str) -> str:
        """
        Helper to get an answer from the VQA module for a given question and image.
        Implements caching and retries with exponential backoff.
        
        Args:
            image: The input image (PIL Image expected).
            question: The question string.
        
        Returns:
            The answer string from the VQA module (empty string if retrieval fails).
        """
        local_logger = logging.getLogger(__name__)
        max_retries: int = self.vqa_max_retries
        delay: float = self.vqa_initial_delay
        attempt: int = 0

        # Create a cache key based on the question and a hash of the image.
        try:
            image_bytes: bytes = image.tobytes()
        except Exception as exc:
            image_bytes = str(hash(image))
        cache_key = generate_api_cache_key((question, image_bytes))
        cached_response: Optional[str] = get_cached_response(cache_key)
        if cached_response is not None:
            local_logger.info(f"VQA answer retrieved from cache for question: '{question}'.")
            return cached_response

        answer: str = ""
        while attempt < max_retries:
            try:
                answer = self.vqa.answer_question(image, question)
                if answer:
                    cache_api_response(cache_key, answer)
                    return answer
            except Exception as exc:
                local_logger.error(f"VQA answer retrieval failed on attempt {attempt+1}/{max_retries} for question: '{question}'. Error: {exc}")
            attempt += 1
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        local_logger.error(f"VQA answer retrieval failed after {max_retries} attempts for question: '{question}'. Returning empty answer.")
        return answer

    def detect_sample(self, image: Any, label: str) -> Dict[str, Any]:
        """
        Detects whether a given sample (image, label) is dirty by:
          1. Generating general and label-specific questions via the VQG module.
          2. Obtaining answers for each question via the VQA module (with caching and retries).
          3. Evaluating each answer using the VAE module.
          4. Computing a vote-based matching score; if below the configured threshold, the sample is marked as dirty.
        
        Args:
            image: The input image (expected as a PIL Image).
            label: The associated label (string).
        
        Returns:
            A dictionary with:
                - "is_dirty": bool (True if the matching score is below the threshold)
                - "score": float (the vote-based matching score)
                - "details": List[Dict[str, Any]] providing diagnostic information per question.
        """
        local_logger = logging.getLogger(__name__)
        diagnostics: List[Dict[str, Any]] = []

        # (1) QUESTION GENERATION
        general_questions: List[str] = []
        label_specific_questions: List[str] = []

        # Generate general questions with retry logic
        retry = 0
        while retry < self.max_retries:
            try:
                general_questions = self.vqg.generate_general_questions(label)
            except Exception as exc:
                local_logger.error(f"Error generating general questions: {exc}")
                general_questions = []
            if len(general_questions) == self.general_q_count:
                break
            else:
                local_logger.warning(
                    f"General questions count mismatch for label '{label}'. Expected {self.general_q_count}, got {len(general_questions)}. Retrying ({retry+1}/{self.max_retries})."
                )
                retry += 1
        if len(general_questions) != self.general_q_count:
            local_logger.warning("Proceeding with available general questions despite count mismatch.")

        # Generate label-specific questions with retry logic
        retry = 0
        while retry < self.max_retries:
            try:
                label_specific_questions = self.vqg.generate_label_specific_questions(label, self.dataset_id)
            except Exception as exc:
                local_logger.error(f"Error generating label-specific questions: {exc}")
                label_specific_questions = []
            if len(label_specific_questions) == self.label_specific_q_count:
                break
            else:
                local_logger.warning(
                    f"Label-specific questions count mismatch for label '{label}', dataset '{self.dataset_id}'. Expected {self.label_specific_q_count}, got {len(label_specific_questions)}. Retrying ({retry+1}/{self.max_retries})."
                )
                retry += 1
        if len(label_specific_questions) != self.label_specific_q_count:
            local_logger.warning("Proceeding with available label-specific questions despite count mismatch.")

        # Combine questions with type annotation.
        combined_questions: List[Dict[str, str]] = []
        for q in general_questions:
            combined_questions.append({"question": q, "type": "general"})
        for q in label_specific_questions:
            combined_questions.append({"question": q, "type": "label-specific"})

        if not combined_questions:
            local_logger.error("No questions generated; sample cannot be evaluated. Marking sample as dirty by default.")
            return {"is_dirty": True, "score": 0.0, "details": diagnostics}

        # (2) ANSWER ACQUISITION & (3) ANSWER EVALUATION VIA VAE
        positive_votes: int = 0
        for q_item in combined_questions:
            q_text: str = q_item.get("question", "")
            q_type: str = q_item.get("type", "general")
            # Retrieve answer using VQA with retry and caching
            answer_text: str = self._get_vqa_answer(image, q_text)
            # Set expected answer; for both types we assume "yes" is expected.
            expected_value: str = "yes"
            try:
                evaluation: bool = self.vae.evaluate_answer(answer_text, expected_value, q_type)
            except Exception as exc:
                local_logger.error(f"Error evaluating answer for question '{q_text}': {exc}")
                evaluation = False
            if evaluation:
                positive_votes += 1
            diagnostics.append({
                "question": q_text,
                "question_type": q_type,
                "answer": answer_text,
                "evaluation": evaluation
            })

        total_questions: int = len(combined_questions)
        matching_score: float = positive_votes / total_questions if total_questions > 0 else 0.0
        is_dirty: bool = matching_score < self.threshold
        decision_str: str = "dirty" if is_dirty else "clean"

        local_logger.info(f"Detection result for label '{label}': {decision_str} with matching score {matching_score:.2f} "
                          f"({positive_votes}/{total_questions} positive evaluations).")
        return {"is_dirty": is_dirty, "score": matching_score, "details": diagnostics}


# Optional test stub for module verification.
if __name__ == "__main__":
    import sys
    from PIL import Image
    from utils import load_config, init_seed

    # Dummy implementations for testing purposes.
    class DummyVQG:
        def __init__(self, cfg=None):
            self.config = cfg
        def generate_general_questions(self, label: str) -> List[str]:
            # For testing, return exactly 2 general questions.
            return [f"General question 1 for {label}", f"General question 2 for {label}"]
        def generate_label_specific_questions(self, label: str, dataset_id: str) -> List[str]:
            # For "CIFAR-10", return 4 label-specific questions.
            count = 4 if dataset_id == "CIFAR-10" else 6
            return [f"Label-specific question {i+1} for {label}" for i in range(count)]

    class DummyVQA:
        def answer_question(self, image: Any, question: str) -> str:
            # Always answer "yes" for testing.
            return "yes"
    
    class DummyVAE:
        def evaluate_answer(self, answer: str, expected: str, question_type: str) -> bool:
            # Evaluate to True if "yes" is found in the answer (case-insensitive).
            return "yes" in answer.lower()

    # Load configuration and initialize seed.
    config = load_config("config.yaml")
    init_seed(config.get("seed", 42))

    dummy_vqg = DummyVQG(config)
    dummy_vqa = DummyVQA()
    dummy_vae = DummyVAE()

    # Create detector instance for dataset "CIFAR-10" using threshold from config.
    detection_threshold = config.get("detection", {}).get("threshold", 0.5)
    detector = VDCDetector(dummy_vqg, dummy_vqa, dummy_vae, threshold=detection_threshold, dataset_id="CIFAR-10", config=config)

    # Create a dummy test image of size 32x32.
    test_image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    test_label = "cat"
    result = detector.detect_sample(test_image, test_label)
    print("Detection result:", result)
