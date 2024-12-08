from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib
import os
import cv2


def train_digits():
    """
    Train a Linear Support Vector Classifier (SVC) model on a dataset of digits images using
    Histogram of Oriented Gradients (HOG) features.

    The images are stored in a folder structure where each subfolder corresponds to a digit class
    and contains images of that digit. The model is saved as 'hog_model_digits.npy'.
    """
    images = []
    labels = []

    # Get all the image folder paths
    image_paths = os.listdir("./digits_dataset")

    for path in image_paths:
        # Get all the image names
        all_images = os.listdir(f"./digits_dataset/{path}")

        # Iterate over the image names, get the label
        for image in all_images:
            image_path = f"./digits_dataset/{path}/{image}"
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))

            # Get the HOG descriptor for the image
            hog_desc = hog(image, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # Update the data and labels
            images.append(hog_desc)
            labels.append(path)

    # Train Linear SVC
    print('Training on training images...')
    svm_model = LinearSVC(random_state=42, tol=1e-5)
    svm_model.fit(images, labels)

    # Save the model
    joblib.dump(svm_model, "hog_model_digits.npy")


def get_prediction(image):
    """
    Predict the label of a digit image using the pre-trained LinearSVC model with HOG features.

    Parameters:
    - image: A grayscale image (as a numpy array) of a digit to be classified.

    Returns:
    - The predicted label of the digit as a string.
    """
    model_filename = "hog_model_digits.npy"

    # Check if the model exists
    if not os.path.exists(model_filename):
        print("Model not found. Training the model...")
        train_digits()  # Train the model if it does not exist
        print("Model trained successfully.")

    # Load the pre-trained model
    hog_model = joblib.load("hog_model_digits.npy")

    resized_image = cv2.resize(image, (28, 28))
    # Get the HOG descriptor for the test image
    hog_desc, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    # Prediction
    pred = hog_model.predict(hog_desc.reshape(1, -1))[0]

    return pred.title()
