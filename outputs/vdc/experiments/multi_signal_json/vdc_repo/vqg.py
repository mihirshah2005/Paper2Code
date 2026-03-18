"""
vqg.py

This module implements the VQG (Visual Question Generation) class for the VDC project.
It generates two types of questions:
  - General Questions using predefined templates.
  - Label-Specific Questions using ChatGPT (gpt-3.5-turbo) via the OpenAI API.

The class is initialized with default templates, prompt template for label-specific
questions, expected counts derived from configuration, and it uses an internal cache
to avoid redundant API calls. Robust error handling is implemented with retries and
fallback strategies for API failures and unexpected response formats.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import openai

from utils import (
    get_cached_api_response,
    set_cached_api_response,
    generate_cache_key,
    clean_text,
)

# Set up module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VQG:
    """
    Visual Question Generation class for generating general and label-specific questions.
    
    Attributes:
        general_templates (List[str]): Predefined general question templates.
        base_label_prompt (str): Template for generating label-specific questions.
        general_questions_count (int): Number of general questions to generate (default 2).
        label_specific_count (int): Number of label-specific questions to generate (default 4).
        max_retries (int): Maximum number of retries for API calls.
        retry_delay (int): Base delay (in seconds) for retry attempts.
        openai_model (str): OpenAI model used for generating label-specific questions.
        cache (Dict[str, List[str]]): Internal cache to store generated label-specific questions.
    """
    def __init__(
        self,
        general_templates: Optional[List[str]] = None,
        base_label_prompt: Optional[str] = None,
        general_questions_count: int = 2,
        label_specific_count: int = 4,
        max_retries: int = 3,
        retry_delay: int = 2,
        openai_model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the VQG module with the provided templates and configuration.
        
        Args:
            general_templates (Optional[List[str]]): List of general question templates.
                Defaults to ["Describe the image briefly.", "Describe the image in detail."].
            base_label_prompt (Optional[str]): Template for label-specific question generation.
                Defaults to a prompt instructing to return a JSON array of questions.
            general_questions_count (int): Number of general questions to select. Default is 2.
            label_specific_count (int): Number of label-specific questions to generate. Default is 4.
            max_retries (int): Maximum count of API retry attempts. Default is 3.
            retry_delay (int): Base delay between API retries in seconds. Default is 2.
            openai_model (str): OpenAI model to use. Default is "gpt-3.5-turbo".
            openai_api_key (Optional[str]): API key for OpenAI. If provided, it is set for openai.
        """
        # Set general templates; default to two common description questions.
        self.general_templates: List[str] = general_templates if general_templates is not None else [
            "Describe the image briefly.",
            "Describe the image in detail."
        ]
        self.general_questions_count: int = general_questions_count

        # Set prompt for generating label-specific questions.
        self.base_label_prompt: str = base_label_prompt if base_label_prompt is not None else (
            "For the label \"{label}\", please generate exactly {n} insightful visual questions "
            "that would help determine if an image correctly depicts a \"{label}\". "
            "Return your answer strictly as a JSON array of strings with no additional text."
        )
        self.label_specific_count: int = label_specific_count

        self.max_retries: int = max_retries
        self.retry_delay: int = retry_delay
        self.openai_model: str = openai_model

        # Set OpenAI API key if provided.
        if openai_api_key:
            openai.api_key = openai_api_key

        # Internal cache for label-specific questions to avoid redundant API calls.
        self.cache: Dict[str, List[str]] = {}

        logger.info(f"Initialized VQG with general_questions_count={self.general_questions_count}, "
                    f"label_specific_count={self.label_specific_count}, max_retries={self.max_retries}, "
                    f"retry_delay={self.retry_delay}, openai_model={self.openai_model}")

    def generate_general_questions(self, label: str) -> List[str]:
        """
        Generates a list of general questions based on predefined templates.
        The label parameter is available for potential future personalization.
        
        Args:
            label (str): The associated label (unused in current implementation).
        
        Returns:
            List[str]: A list containing the general questions.
        """
        # Select the first 'general_questions_count' templates.
        selected_questions = self.general_templates[: self.general_questions_count]
        logger.info(f"Generated general questions for label '{label}': {selected_questions}")
        return selected_questions

    def generate_label_specific_questions(self, label: str) -> List[str]:
        """
        Generates a list of label-specific questions using the ChatGPT API.
        Implements retries, error handling, and a fallback strategy if the API call
        fails or returns insufficient questions.
        
        Args:
            label (str): The label for which to generate questions.
        
        Returns:
            List[str]: A list of label-specific questions of length equal to label_specific_count.
        """
        # Check internal cache first.
        if label in self.cache:
            logger.info(f"Label-specific questions for '{label}' found in cache.")
            return self.cache[label]

        expected_count: int = self.label_specific_count
        prompt: str = self.base_label_prompt.format(label=label, n=expected_count)
        logger.info(f"Constructed prompt for label-specific questions: {prompt}")

        # Generate a unique cache key for this prompt.
        cache_key = generate_cache_key(prompt=prompt)

        # Check global API response cache as well.
        cached_response = get_cached_api_response(cache_key)
        if cached_response:
            logger.info(f"Found cached API response for key: {cache_key}")
            questions = self._parse_questions(cached_response, expected_count, label)
            if len(questions) >= expected_count:
                self.cache[label] = questions[:expected_count]
                return self.cache[label]

        # Attempt to call the ChatGPT API with retries.
        response_text: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt} for API call to generate questions for label '{label}'.")
                messages = [
                    {"role": "system", "content": "You are an expert assistant specialized in generating visual questions."},
                    {"role": "user", "content": prompt}
                ]
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512,
                    n=1,
                )
                response_text = response['choices'][0]['message']['content']
                logger.info(f"Received API response for label '{label}': {response_text}")
                # Cache the raw API response.
                set_cached_api_response(cache_key, response_text)
                # Exit loop if successful.
                break
            except Exception as e:
                logger.error(f"Error during ChatGPT API call on attempt {attempt} for label '{label}': {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.critical(f"Max retries reached for label '{label}'. Using fallback strategy.")
                    response_text = None

        # If API call failed and no response obtained, use fallback.
        if not response_text:
            fallback_questions = self._fallback_questions(label, expected_count)
            self.cache[label] = fallback_questions
            return fallback_questions

        # Parse the API response and validate the questions.
        questions = self._parse_questions(response_text, expected_count, label)
        if len(questions) < expected_count:
            logger.warning(f"Insufficient questions ({len(questions)}) extracted for label '{label}'. "
                           f"Expected {expected_count}. Using fallback to append default questions.")
            # Append fallback questions until the list reaches expected_count.
            fallback = self._fallback_questions(label, expected_count - len(questions))
            questions.extend(fallback)

        # Ensure we only return exactly the expected count.
        final_questions = questions[:expected_count]
        # Cache the final output.
        self.cache[label] = final_questions
        logger.info(f"Final label-specific questions for '{label}': {final_questions}")
        return final_questions

    def _parse_questions(self, response_text: str, expected_count: int, label: str) -> List[str]:
        """
        Attempts to parse questions from the API response using a multi-stage approach:
        1. Try JSON parsing.
        2. Fallback to heuristic regex extraction if JSON parsing fails.
        
        Args:
            response_text (str): The raw API response.
            expected_count (int): The expected number of questions.
            label (str): The label for which questions are generated.
        
        Returns:
            List[str]: A list of extracted questions.
        """
        questions: List[str] = []
        # First, try to parse the response as JSON.
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                # Clean and validate each question.
                for q in parsed:
                    if isinstance(q, str):
                        q_cleaned = clean_text(q)
                        if q_cleaned and q_cleaned.endswith("?") and len(q_cleaned) > 10:
                            questions.append(q_cleaned)
            else:
                logger.warning("JSON parsed object is not a list.")
        except json.JSONDecodeError as json_err:
            logger.warning(f"JSON parsing failed: {json_err}. Falling back to heuristic extraction.")

        # If JSON parsing did not yield enough questions, use regex-based heuristic.
        if len(questions) < expected_count:
            heuristic_questions = self._heuristic_question_extraction(response_text)
            # Merge with already found questions while avoiding duplicates.
            for q in heuristic_questions:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= expected_count:
                    break

        return questions

    def _heuristic_question_extraction(self, text: str) -> List[str]:
        """
        Uses regex heuristics to extract candidate questions from a block of text.
        Looks for sentences ending with '?'.
        
        Args:
            text (str): The text from which to extract questions.
        
        Returns:
            List[str]: A list of candidate questions.
        """
        # Use a regex pattern to find sentences that end with a question mark.
        pattern = r"([A-Z][^?]*\?)"
        candidates = re.findall(pattern, text)
        # Clean candidates and filter out overly short ones.
        extracted = []
        for candidate in candidates:
            candidate_clean = clean_text(candidate)
            if len(candidate_clean) >= 10 and candidate_clean.endswith("?"):
                extracted.append(candidate_clean)
        logger.info(f"Heuristic extraction yielded {len(extracted)} candidate questions.")
        return extracted

    def _fallback_questions(self, label: str, count: int) -> List[str]:
        """
        Provides a fallback list of default questions for a given label.
        
        Args:
            label (str): The label for which to create fallback questions.
            count (int): Number of fallback questions required.
        
        Returns:
            List[str]: A list of default fallback questions.
        """
        # Fallback template: simple question based on the label.
        fallback_template = f"Does the image clearly depict a typical {label}?"
        fallback_list = [fallback_template for _ in range(count)]
        logger.info(f"Using fallback questions for label '{label}': {fallback_list}")
        return fallback_list
