import cv2
import numpy as np
from imutils import contours as imcnts
from paper_extraction import *
from bubble_sheet_correction import *
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Function to segment the student ID from the bubble sheet.
def segment_id(code):
    """
    Segments the student ID from the provided bubble sheet image.

    Args:
        code (numpy.ndarray): The binary image of the bubble sheet.

    Returns:
        list: A list of segmented digits.
    """
    # Find external contours and filter bounding boxes based on size.
    contours, _ = cv2.findContours(code, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_rectangles = [rect for rect in bounding_rectangles if rect[2] * rect[3] > 100]
    bounding_rectangles = sorted(bounding_rectangles, key=lambda x: x[0])

    digits = []
    for rect in bounding_rectangles:
        x, y, w, h = rect
        digit = code[y:y+h, x:x+w]
        digits.append(digit)

    return digits


# Function to extract the bubble code from the paper.
def extract_bubble_code(paper):
    """
    Extracts the bubble code area from the input paper image.

    Args:
        paper (numpy.ndarray): The input paper image.

    Returns:
        numpy.ndarray: The segmented bubble code area.
    """
    # Define new X and Y limits based on the shape of the paper.
    x, y = paper.shape[:2]
    new_x = (x // 3) + 10
    new_y = (y // 2) + 40

    segment = paper[:new_x, :new_y]
    return segment


# Function to extract the student ID code from the paper.
def extract_student_code(paper):
    """
    Extracts the student code from the paper image, performing thresholding and contour-based extraction.

    Args:
        paper (numpy.ndarray): The input paper image.

    Returns:
        numpy.ndarray: The transformed and segmented student code.
    """
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10
    )
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        target_area = 30000
        for contour in contours:
            epsilon_value = 0.01 * cv2.arcLength(contour, True)
            paper_contour = cv2.approxPolyDP(contour, epsilon_value, True)

            if len(paper_contour) == 4 and cv2.contourArea(contour) > 0.2 * target_area:
                code = image_transform(paper, paper_contour.reshape(4, 2))

    gray_code = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
    blurred_code = cv2.GaussianBlur(gray_code, (5, 5), 0.5)
    binary_code = cv2.adaptiveThreshold(blurred_code, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10)
    negative_code = negative_transformation(binary_code)
    eroded_image = cv2.erode(negative_code, np.ones((2, 2), np.uint8), iterations=1)
    dilated_image = cv2.dilate(eroded_image, np.ones((3, 3), np.uint8), iterations=1)

    return dilated_image


# Function to crop the student code from the image.
def crop_code(code):
    """
    Crops the student code region from the image.

    Args:
        code (numpy.ndarray): The binary image of the student code.

    Returns:
        numpy.ndarray: The cropped student code region.
    """
    kernel = np.ones((10, 25), np.uint8)
    dilated_image = cv2.dilate(code, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = code[y:y+h, x:x+w]
    return cropped_image


def preprocess_paper_image(paper):
    """
    Preprocess the paper image for bubble detection.

    Args:
        paper (numpy.ndarray): Input paper image.

    Returns:
        tuple: Grayscale, binary, and negative images.
    """
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 15
    )
    negative_img = negative_transformation(binary_image)
    eroded_image = cv2.erode(negative_img, np.ones((7, 7)), iterations=1)

    return gray_image, binary_image, negative_img, eroded_image


def filter_bubble_contours(all_contours):
    """
    Filter and identify bubble contours from all contours.

    Args:
        all_contours (list): List of contours to filter.

    Returns:
        tuple: Filtered circle contours and their areas.
    """
    circles_contours = []
    areas_of_contours = []

    for contour in all_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon_value = 0.01 * cv2.arcLength(contour, True)
        circle_contour = cv2.approxPolyDP(contour, epsilon_value, True)

        if (0.5 <= aspect_ratio <= 1.5 and
                len(circle_contour) >= 4 and
                cv2.contourArea(contour) > 30 and
                cv2.contourArea(contour) > 1.5 * cv2.arcLength(contour, True)):
            circles_contours.append(contour)
            areas_of_contours.append(cv2.contourArea(contour))

    return circles_contours, areas_of_contours


def filter_consistent_bubbles(circles_contours, areas_of_contours):
    """
    Filter bubbles with consistent area.

    Args:
        circles_contours (list): List of bubble contours.
        areas_of_contours (list): List of corresponding contour areas.

    Returns:
        list: Filtered bubble contours.
    """
    median_circle_area = np.median(areas_of_contours)
    filtered_contours = [
        contour for contour, area in zip(circles_contours, areas_of_contours)
        if abs(area - median_circle_area) <= median_circle_area * 0.1
    ]

    return filtered_contours


def determine_row_structure(sorted_contours):
    """
    Determine the number of questions per row and number of answer choices.

    Args:
        sorted_contours (list): Sorted contours from top to bottom.

    Returns:
        tuple: Number of questions per row and answers per question.
    """
    (_, __, circle_width, ___) = cv2.boundingRect(sorted_contours[0])
    (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])

    first_row = [x_prev]
    for contour in sorted_contours[1:]:
        (x, y, _, _) = cv2.boundingRect(contour)
        if abs(y - y_prev) > 3:
            break
        first_row.append(x)

    first_row.sort()
    questions_number_per_row = 1
    circles_number = len(first_row)
    for i in range(1, circles_number):
        if first_row[i] - first_row[i - 1] > 2.5 * circle_width:
            questions_number_per_row += 1

    answers_number_per_question = circles_number // questions_number_per_row

    logger.debug(f"Questions per row: {questions_number_per_row}")
    logger.debug(f"Answers per question: {answers_number_per_question}")

    return questions_number_per_row, answers_number_per_question


def process_student_answers(sorted_contours, circle_width, eroded_image):
    """
    Process student answers from the sorted contours.

    Args:
        sorted_contours (list): Contours sorted top-to-bottom.
        circle_width (int): Width of a bubble circle.
        eroded_image (numpy.ndarray): Preprocessed image.

    Returns:
        tuple: Student answers and validation array.
    """
    # Log initial contour information
    logger.debug(f"Total sorted contours: {len(sorted_contours)}")

    try:
        (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])
        curr_row = 0
        x_list = [[x_prev, sorted_contours[0]]]

        # Dynamically determine number of questions based on contours
        questions_number_per_row, answers_number_per_question = determine_row_structure(sorted_contours)

        # Calculate total number of questions more dynamically
        number_of_questions = len(sorted_contours) // answers_number_per_question
        logger.debug(f"Calculated number of questions: {number_of_questions}")

        # Adjust array size to match expected questions
        student_answers = np.zeros(number_of_questions, dtype=int)
        chosen_contours = [None] * number_of_questions
        student_answers_validate = np.zeros(number_of_questions, dtype=int)

        for contour in sorted_contours[1:]:
            (x, y, _, _) = cv2.boundingRect(contour)
            if abs(y - y_prev) > 3:
                x_list.sort(key=lambda pair: pair[0])
                question_per_row = 1
                answer = 1
                question_num = curr_row

                for i in range(len(x_list)):
                    if (i - 1 >= 0) and ((x_list[i][0] - x_list[i - 1][0]) > (2.5 * circle_width)):
                        question_num = curr_row + question_per_row
                        question_per_row += 1
                        answer = 1

                    # Add safety check to prevent index out of bounds
                    if question_num < number_of_questions:
                        if get_choice(x_list[i][1], eroded_image) == 1:
                            student_answers[question_num] = answer
                            chosen_contours[question_num] = x_list[i][1]
                            student_answers_validate[question_num] += 1
                    else:
                        logger.warning(f"Skipping question {question_num} - out of bounds")

                    answer += 1

                curr_row += 1
                x_list = [[x, contour]]
            else:
                x_list.append([x, contour])

        return student_answers, chosen_contours, student_answers_validate

    except Exception as e:
        logger.error(f"Error processing student answers: {e}")
        logger.error(f"Contour details: {sorted_contours}")
        raise


def get_student_bubble_code(paper):
    """
    Extract student answers from a bubble sheet.

    Args:
        paper (numpy.ndarray): Input paper image.

    Returns:
        tuple: Student answers and context information.
    """
    # Preprocess image
    _, binary_image, negative_img, eroded_image = preprocess_paper_image(paper)

    # Find and filter contours
    all_contours, _ = cv2.findContours(negative_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circles_contours, areas_of_contours = filter_bubble_contours(all_contours)

    # Log contour information
    logger.debug(f"Total contours found: {len(all_contours)}")
    logger.debug(f"Filtered bubble contours: {len(circles_contours)}")

    # Further filter bubbles
    filtered_contours = filter_consistent_bubbles(circles_contours, areas_of_contours)

    # Prepare result visualization
    contoured_paper = paper.copy()

    # Sort contours and process answers
    sorted_contours, _ = imcnts.sort_contours(filtered_contours, method='top-to-bottom')
    (_, __, circle_width, ___) = cv2.boundingRect(sorted_contours[0])

    # Get student answers
    student_answers, _, _ = process_student_answers(
        sorted_contours, circle_width, eroded_image
    )

    return student_answers
