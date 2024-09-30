import numpy as np
from digit_classification_interface import DigitClassificationInterface

class RandomModel(DigitClassificationInterface):
    """
    Random model that predicts a random digit class.
    """

    def __init__(self):
        pass  # No initialization required for the random model

    def predict(self, image: np.ndarray) -> int:
        """
        Predicts a random digit class for a given image.

        Parameters:
        image (np.ndarray): The input image array of shape (28, 28, 1).

        Returns:
        int: A random digit class (0-9).
        """
        # Optionally, perform center cropping to (10, 10)
        center_crop = image[9:19, 9:19, 0]  # Extract the center 10x10 region

        # Generate a random digit between 0 and 9
        predicted_class = np.random.randint(0, 10)
        return predicted_class

    def train(self, X, y):
        """
        Training function for the Random model.
        Raises NotImplementedError as training is not required.
        """
        raise NotImplementedError("Training function is not implemented for RandomModel.")
