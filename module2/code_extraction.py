import cv2
import numpy as np
from imutils import contours as imcnts
from .pape_extraction import *
from .bubble_sheet_correction import *
from .train_digits import *


# Function to segment and extract individual digits from a given binary code image
def segment_id(code):
    """
    Segments and extracts individual digits from the given binary code image.

    Parameters:
        code (numpy.ndarray): Binary image containing digits.

    Returns:
        list[numpy.ndarray]: List of segmented digit images.
    """
    contours, _ = cv2.findContours(code, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rectangles = [
        cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100
    ]
    bounding_rectangles.sort(key=lambda x: x[0])  # Sort by x-coordinate
    digits = [code[y : y + h, x : x + w] for x, y, w, h in bounding_rectangles]
    return digits


# Function to extract the bubble code region from the answer sheet
def extract_bubble_code(paper):
    """
    Extracts the bubble code region from the answer sheet.

    Parameters:
        paper (numpy.ndarray): Input paper image.

    Returns:
        numpy.ndarray: Extracted bubble code region.
    """
    height, width = paper.shape[:2]
    return paper[: height // 3 + 10, : width // 2 + 40]


# Function to extract the student's code region
def extract_student_code(paper):
    """
    Extracts and processes the student's code region.

    Parameters:
        paper (numpy.ndarray): Input paper image.

    Returns:
        numpy.ndarray: Processed binary code image.
    """
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10
    )
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 30000 * 0.2:
            code_region = apply_perspective_transform(paper, approx.reshape(4, 2))
            break

    gray_code = cv2.cvtColor(code_region, cv2.COLOR_BGR2GRAY)
    blurred_code = cv2.GaussianBlur(gray_code, (5, 5), 0.5)
    binary_code = cv2.adaptiveThreshold(
        blurred_code, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10
    )
    negative_code = 255 - binary_code
    processed_code = cv2.dilate(
        cv2.erode(negative_code, np.ones((2, 2), np.uint8), iterations=1),
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    return processed_code


# Function to crop the code region of interest
def crop_code(code):
    """
    Crops the relevant region of the code.

    Parameters:
        code (numpy.ndarray): Input binary code image.

    Returns:
        numpy.ndarray: Cropped region.
    """
    dilated = cv2.dilate(code, np.ones((10, 25), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return code[y : y + h, x : x + w]


# Function to extract and process the student's bubble answers
def get_student_bubble_code(paper):
    """
    Extracts the bubble answer code from the student's sheet.

    Parameters:
        paper (numpy.ndarray): Input paper image.

    Returns:
        tuple: Annotated paper with contours and extracted student answers.
    """
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 15
    )
    negative_image = 255 - binary_image
    eroded_image = cv2.erode(negative_image, np.ones((7, 7), np.uint8), iterations=1)

    # Extract bubble contours
    contours, _ = cv2.findContours(
        negative_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bubble_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if (
            0.5 <= aspect_ratio <= 1.5
            and len(approx) >= 4
            and 30 < cv2.contourArea(contour) < 1.5 * cv2.arcLength(contour, True)
        ):
            bubble_contours.append(contour)

    # Sort and filter valid bubble contours
    sorted_contours, _ = imcnts.sort_contours(bubble_contours, method="top-to-bottom")
    median_area = np.median([cv2.contourArea(c) for c in sorted_contours])
    valid_contours = [
        c
        for c in sorted_contours
        if abs(cv2.contourArea(c) - median_area) <= 0.1 * median_area
    ]

    # Determine student answers
    annotated_paper = paper.copy()
    student_answers = []
    for contour in valid_contours:
        if is_bubble_filled(contour, eroded_image):
            student_answers.append(contour)
            cv2.drawContours(annotated_paper, [contour], -1, (255, 0, 0), 2)

    return annotated_paper, student_answers


# Function to predict digits from segmented images
def get_code_prediction(digits):
    """
    Predicts digits from segmented digit images.

    Parameters:
        digits (list[numpy.ndarray]): List of digit images.

    Returns:
        list[int]: Predicted digits.
    """
    predictions = []
    for digit in digits:
        eroded = cv2.erode(digit, np.ones((2, 2), np.uint8), iterations=1)
        resized = cv2.resize(eroded, (28, 28))
        predictions.append(predict_digit(resized))
    return predictions
