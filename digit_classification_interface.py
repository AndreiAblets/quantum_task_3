from abc import ABC, abstractmethod
import numpy as np

class DigitClassificationInterface(ABC):
    """
    Interface for digit classification models.
    All models should inherit from this class and implement the predict method.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predicts the digit class for a given image.

        Parameters:
        image (np.ndarray): The input image array of shape (28, 28, 1).

        Returns:
        int: The predicted digit class (0-9).
        """
        pass

    @abstractmethod
    def train(self, X, y):
        """
        Training function for the model.
        Since training is not required, this method will raise a NotImplementedError.

        Parameters:
        X: Training data features.
        y: Training data labels.
        """
        raise NotImplementedError("Training function is not implemented.")
