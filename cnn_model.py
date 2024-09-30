import numpy as np
from digit_classification_interface import DigitClassificationInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class CNNModel(DigitClassificationInterface):
    """
    Convolutional Neural Network model for digit classification.
    """

    def __init__(self):
        # Define the CNN architecture
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        print("CNN model initialized with random weights.")

    def predict(self, image: np.ndarray) -> int:
        """
        Predicts the digit class for a given image using the CNN model.

        Parameters:
        image (np.ndarray): The input image tensor of shape (28, 28, 1).

        Returns:
        int: The predicted digit class (0-9).
        """
        # Preprocess the image
        image = image.reshape(1, 28, 28, 1) / 255.0

        # Make prediction
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return int(predicted_class)

    def train(self, X, y):
        """
        Training function for the CNN model.
        Raises NotImplementedError as training is not required.
        """
        raise NotImplementedError("Training function is not implemented for CNNModel.")
