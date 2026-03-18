"""
detection.py

This module implements the VDCDetector class which integrates three core modules:
    - VQG: Visual Question Generation
    - VQA: Visual Question Answering
    - VAE: Visual Answer Evaluation

The VDCDetector is responsible for determining whether an input sample (image and label)
is "dirty" (i.e., exhibits visual-linguistic inconsistency) by:
  1. Generating a set of questions (both general and label‐specific),
  2. Obtaining corresponding answers using a multimodal model (VQA module),
  3. Evaluating each answer via the VAE module,
  4. Computing a vote-based matching score and comparing it against the threshold to flag the sample.

For label-specific questions, the expected answer is pre-defined as "yes". For general questions,
the expected answer is constructed as "the image is about {label}" based on the provided label.
The matching score is calculated by dividing the number of correct evaluations by the total questions.
If the score is less than the threshold (default 0.5), the sample is marked as dirty.

Configuration parameters (e.g., threshold) are sourced from config.yaml via the central configuration
logic in the project.
"""

import logging
from typing import Any, Dict, List, Tuple

# Import the necessary modules from the project.
from vqg import VQG
from vqa import VQA
from vae import VAE
from utils import clean_text

# Initialize a logger for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Setup a console handler if not already present.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)


class VDCDetector:
    """
    Versatile Data Cleanser (VDC) Detector integrating VQG, VQA, and VAE modules.

    Attributes:
        vqg (VQG): Instance of Visual Question Generation module.
        vqa (VQA): Instance of Visual Question Answering module.
        vae (VAE): Instance of Visual Answer Evaluation module.
        threshold (float): Matching score threshold below which a sample is considered dirty.
    """

    def __init__(self, vqg: VQG, vqa: VQA, vae: VAE, threshold: float = 0.5) -> None:
        """
        Initialize the VDCDetector with instances of VQG, VQA, and VAE modules and threshold.

        Args:
            vqg (VQG): Visual Question Generation instance.
            vqa (VQA): Visual Question Answering instance.
            vae (VAE): Visual Answer Evaluation instance.
            threshold (float, optional): The matching score threshold (default is 0.5).
        """
        self.vqg = vqg
        self.vqa = vqa
        self.vae = vae
        self.threshold: float = threshold
        logger.info("VDCDetector initialized with threshold set to %.2f", self.threshold)

    def detect_sample(self, image: Any, label: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect whether the given sample (image and label) is dirty by computing a matching score.

        Detection Process:
          1. Generate two sets of questions using VQG:
             - General questions (using fixed templates)
             - Label-specific questions (generated via ChatGPT)
          2. For label-specific questions, embed the expected answer "yes".
             For general questions, construct an expected answer as "the image is about {label}".
          3. For each question in the combined list:
             a. Use VQA to answer the question based on the input image.
             b. Evaluate the answer using VAE.evaluate_answer method with the expected answer
                and question type ("general" or "label-specific").
          4. Compute the matching score as (number of correct evaluations) / (total number of questions).
          5. If the matching score is below the threshold, mark the sample as dirty (return True).

        Args:
            image (Any): The input image (PIL Image or numpy array).
            label (str): The label associated with the image.

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - bool: True if the sample is considered dirty, False if clean.
                - Dict[str, Any]: Details containing matching score, total questions, number of correct votes,
                                  and per-question evaluation details.
        """
        evaluation_details: List[Dict[str, Any]] = []

        # Step 1: Generate questions using VQG.
        # Generate general questions.
        general_questions: List[str] = self.vqg.generate_general_questions(label)
        # For each general question, we set the question type as "general"
        # and set expected answer as "the image is about <label>".
        general_items: List[Dict[str, Any]] = [
            {
                "question": q,
                "question_type": "general",
                "expected_answer": f"the image is about {label}"
            }
            for q in general_questions
        ]

        # Generate label-specific questions.
        label_specific_questions: List[str] = self.vqg.generate_label_specific_questions(label)
        # For label-specific, expected answer is predetermined as "yes" (deterministic matching).
        label_specific_items: List[Dict[str, Any]] = [
            {
                "question": q,
                "question_type": "label-specific",
                "expected_answer": "yes"
            }
            for q in label_specific_questions
        ]

        # Merge both sets.
        all_questions: List[Dict[str, Any]] = general_items + label_specific_items
        total_questions: int = len(all_questions)
        logger.info("Total number of questions generated for label '%s': %d", label, total_questions)

        # Step 2: Obtain answers for each question using VQA.
        num_correct: int = 0
        for idx, q_item in enumerate(all_questions):
            question_text: str = q_item["question"]
            question_type: str = q_item["question_type"]
            expected_answer: str = q_item["expected_answer"]

            logger.info("Processing question %d/%d: [%s] %s",
                        idx + 1, total_questions, question_type, question_text)

            # Get the answer from VQA.
            try:
                answer: str = self.vqa.answer_question(image, question_text)
            except Exception as e:
                logger.error("Error obtaining answer for question '%s': %s", question_text, str(e))
                answer = ""

            # Evaluate the answer using VAE.
            try:
                evaluation: bool = self.vae.evaluate_answer(answer, expected_answer, question_type)
            except Exception as e:
                logger.error("Error evaluating answer '%s' for question '%s': %s",
                             answer, question_text, str(e))
                evaluation = False

            # Count correct evaluations.
            if evaluation:
                num_correct += 1

            # Log individual question details.
            evaluation_details.append({
                "question": question_text,
                "question_type": question_type,
                "expected_answer": expected_answer,
                "obtained_answer": answer,
                "evaluation": evaluation
            })
            logger.info("Question evaluation result: %s", evaluation_details[-1])

        # Step 3: Calculate matching score.
        matching_score: float = num_correct / total_questions if total_questions > 0 else 0.0
        logger.info("Computed matching score: %.2f (%d correct out of %d questions)", 
                    matching_score, num_correct, total_questions)

        # Step 4: Decision based on threshold.
        is_dirty: bool = matching_score < self.threshold
        if is_dirty:
            logger.info("Sample marked as DIRTY (matching score %.2f below threshold %.2f).", 
                        matching_score, self.threshold)
        else:
            logger.info("Sample marked as CLEAN (matching score %.2f meets/exceeds threshold %.2f).", 
                        matching_score, self.threshold)

        # Return decision along with evaluation details.
        details: Dict[str, Any] = {
            "matching_score": matching_score,
            "total_questions": total_questions,
            "num_correct": num_correct,
            "evaluations": evaluation_details
        }
        return is_dirty, details
