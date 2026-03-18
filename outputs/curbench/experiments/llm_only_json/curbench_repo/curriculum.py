"""
curriculum.py

This module defines the CurriculumScheduler abstract base class and a concrete implementation,
SimpleCurriculumScheduler, that implements a basic curriculum learning strategy.
The strategy adjusts a dynamic threshold based on training progress (epochs) and computes a weight 
via a sigmoid function of the batch loss relative to the threshold. The returned dictionary from 
update_schedule contains the "weight" along with additional information (batch count, current epoch, threshold).

The design supports both per-epoch and per-batch updates:
    - When update_schedule is called with an epoch different from the internally stored current_epoch,
      the scheduler resets its per-epoch counter and updates its threshold.
    - Otherwise, it increments the batch counter and computes a weight for the current batch.
      
Curriculum hyperparameters (warmup_epochs, schedule_epochs, growth_rate) are read from 
the configuration dictionary (provided by config.yaml via config.py). If any are missing (None), 
default values are used with a warning:
    - warmup_epochs: default 5
    - schedule_epochs: default 10
    - growth_rate: default 0.1

The returned dictionary from update_schedule is expected to be used by the training loop to adjust 
loss weighting or guide sample selection based on the evolving training dynamics.
      
Note: This module assumes that the configuration dictionary is passed in from a higher-level module 
(e.g., Main or Trainer) and does not depend on global variables.
"""

import math
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict

# Set up a module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CurriculumScheduler(ABC):
    """
    Abstract base class for curriculum learning strategies.

    Attributes:
        warmup_epochs (int): Number of epochs considered as warm-up. During warmup, the threshold is fixed.
        schedule_epochs (int): Epoch interval to update or adjust internal state (default provided but not actively used).
        growth_rate (float): The rate at which the threshold increases after the warmup period.
        current_epoch (int): The last epoch for which the scheduler was updated.
        batch_count (int): Counter for the number of batches processed in the current epoch.
        threshold (float): A dynamic threshold computed per epoch used to modulate the weight.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the CurriculumScheduler with curriculum hyperparameters obtained from config.
        Uses default values if any parameters are missing (None) and emits warnings.

        Args:
            config (dict): The configuration dictionary (e.g., from config.yaml via config.py).
        """
        curriculum_config = config.get("curriculum", {})

        temp_warmup: Any = curriculum_config.get("warmup_epochs", None)
        if temp_warmup is None:
            warnings.warn("warmup_epochs not specified in configuration; using default value 5.")
            self.warmup_epochs: int = 5
        else:
            self.warmup_epochs = int(temp_warmup)

        temp_schedule: Any = curriculum_config.get("schedule_epochs", None)
        if temp_schedule is None:
            warnings.warn("schedule_epochs not specified in configuration; using default value 10.")
            self.schedule_epochs: int = 10
        else:
            self.schedule_epochs = int(temp_schedule)

        temp_growth: Any = curriculum_config.get("growth_rate", None)
        if temp_growth is None:
            warnings.warn("growth_rate not specified in configuration; using default value 0.1.")
            self.growth_rate: float = 0.1
        else:
            self.growth_rate = float(temp_growth)

        self.current_epoch: int = -1  # Indicates that no batch has been processed yet.
        self.batch_count: int = 0
        self.threshold: float = 1.0  # Initial threshold value.

        logger.info(
            f"CurriculumScheduler initialized with warmup_epochs={self.warmup_epochs}, "
            f"schedule_epochs={self.schedule_epochs}, growth_rate={self.growth_rate}"
        )

    @abstractmethod
    def update_schedule(self, loss: float, epoch: int) -> Dict[str, Any]:
        """
        Abstract method to update the curriculum learning schedule.
        It must be called at every training batch with the current batch loss and epoch number. 

        The scheduler distinguishes between new epoch and batch updates:
            - If epoch != current_epoch, reset batch_count and update threshold.
            - Otherwise, increment the batch_count.

        Returns:
            dict: A dictionary containing at least:
                "weight": float - the computed weight for the current batch.
                Additional keys may include "batch_count", "current_epoch", "threshold" for debugging.
        """
        pass


class SimpleCurriculumScheduler(CurriculumScheduler):
    """
    A simple concrete implementation of CurriculumScheduler using a sigmoid-based
    function to compute a weight from the current batch loss and a dynamic threshold.
    
    The dynamic threshold is set as follows:
        - If the current epoch is less than warmup_epochs, the threshold is fixed at 1.0.
        - Otherwise, the threshold is computed as: 1.0 + growth_rate * (epoch - warmup_epochs)
    
    The weight is computed with:
        weight = 1 / (1 + exp(loss - threshold))
    
    This formulation returns a weight in (0, 1) where:
        - If loss is much lower than threshold, weight ~ 1 (sample considered “easy”).
        - If loss is around threshold, weight ~ 0.5.
        - If loss is much greater than threshold, weight ~ 0 (sample considered “hard”).
    """

    def update_schedule(self, loss: float, epoch: int) -> Dict[str, Any]:
        """
        Updates the curriculum schedule based on the current loss and epoch.
        Resets the batch counter when a new epoch is detected.

        Args:
            loss (float): The average loss for the current training batch.
            epoch (int): The current epoch number provided by the training loop.

        Returns:
            dict: A dictionary with keys:
                "weight": float - the computed weight for use in loss reweighting or sample selection.
                "batch_count": int - number of batches processed in the current epoch.
                "current_epoch": int - synchronized current epoch.
                "threshold": float - current dynamic threshold value.
        """
        if epoch != self.current_epoch:
            # New epoch detected; reset batch counter and update threshold.
            self.current_epoch = epoch
            self.batch_count = 1  # Start counting this first batch.
            if epoch < self.warmup_epochs:
                self.threshold = 1.0
            else:
                self.threshold = 1.0 + self.growth_rate * (epoch - self.warmup_epochs)
            logger.info(
                f"New epoch detected: {epoch}. Resetting batch count and updating threshold to {self.threshold:.4f}"
            )
        else:
            self.batch_count += 1

        # Compute weight using a sigmoid function:
        # When loss equals threshold, weight is 0.5.
        try:
            exponent_value: float = loss - self.threshold
            weight: float = 1.0 / (1.0 + math.exp(exponent_value))
        except OverflowError:
            # In case the exponent is too large in magnitude, decide weight based on the sign.
            weight = 0.0 if loss - self.threshold > 0 else 1.0

        return {
            "weight": weight,
            "batch_count": self.batch_count,
            "current_epoch": self.current_epoch,
            "threshold": self.threshold
        }
