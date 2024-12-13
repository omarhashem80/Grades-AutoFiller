from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib
import os
import cv2


def train_digit_classifier():
    """
    Trains a Linear SVC classifier on a dataset of digit images using HOG features.
    """
    hog_features = []
    digit_labels = []

    # Get all folder paths containing digit images
    dataset_paths = os.listdir("./digitsDataSet")

    for digit_folder in dataset_paths:
        # Get all image file names in the folder
        image_files = os.listdir(f"./digitsDataSet/{digit_folder}")

        # Iterate over the image files and extract their labels
        for image_file in image_files:
            image_path = f"./digitsDataSet/{digit_folder}/{image_file}"
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))

            # Compute the HOG descriptor for the image
            hog_descriptor = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                transform_sqrt=True,
                block_norm='L2-Hys'
            )

            # Update feature list and labels
            hog_features.append(hog_descriptor)
            digit_labels.append(digit_folder)

    # Train the Linear SVC model
    print('Training the digit classifier...')
    digit_classifier = LinearSVC(random_state=42, tol=1e-5)
    digit_classifier.fit(hog_features, digit_labels)

    # Save the trained model
    joblib.dump(digit_classifier, "hog_digit_classifier_model.npy")


def predict_digit(image):
    """
    Predicts the digit in a given image using the pre-trained Linear SVC model.

    Args:
        image (numpy.ndarray): Input image containing a digit.

    Returns:
        str: Predicted digit label as a string.
    """
    model_path = "hog_digit_classifier_model.npy"

    # Load the pre-trained model
    digit_classifier = joblib.load(model_path)

    # Preprocess the input image
    resized_image = cv2.resize(image, (28, 28))

    # Compute the HOG descriptor for the input image
    hog_descriptor = hog(
        resized_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm='L2-Hys',
        visualize=False
    )

    # Predict the label for the input image
    predicted_label = digit_classifier.predict(hog_descriptor.reshape(1, -1))[0]

    return predicted_label.title()


# Uncomment to train the model
train_digit_classifier()
