import numpy as np
from digit_classifier import DigitClassifier

def main():
    # Example usage of DigitClassifier

    # Choose the algorithm ('cnn', 'rf', or 'rand')
    algorithm = 'cnn'  # Change to 'rf' or 'rand' as needed

    # Initialize the DigitClassifier
    classifier = DigitClassifier(algorithm=algorithm)

    # Prepare a sample image of shape (28, 28, 1)
    sample_image = np.random.randint(0, 256, (28, 28, 1), dtype=np.uint8)

    # Predict the digit class
    predicted_digit = classifier.predict(sample_image)
    print(f"Predicted Digit: {predicted_digit}")

if __name__ == '__main__':
    main()
