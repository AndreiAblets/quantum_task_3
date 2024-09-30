import numpy as np
from cnn_model import CNNModel
from random_forest_model import RandomForestModel
from random_model import RandomModel

class DigitClassifier:
    """
    DigitClassifier that selects the appropriate model based on the algorithm parameter.
    """

    def __init__(self, algorithm: str):
        """
        Initializes the DigitClassifier with the specified algorithm.

        Parameters:
        algorithm (str): The algorithm to use ('cnn', 'rf', or 'rand').

        Raises:
        ValueError: If an unsupported algorithm is specified.
        """
        if algorithm == 'cnn':
            self.model = CNNModel()
        elif algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from 'cnn', 'rf', or 'rand'.")
        print(f"DigitClassifier initialized with '{algorithm}' algorithm.")

    def predict(self, image: np.ndarray) -> int:
        """
        Predicts the digit class for a given image using the selected algorithm.

        Parameters:
        image (np.ndarray): The input image array of shape (28, 28, 1).

        Returns:
        int: The predicted digit class (0-9).
        """
        return self.model.predict(image)
