import cv2
import numpy as np
from imutils import contours as imcnts


def negative_transformation(image):
    """
    Applies a negative transformation to the given image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Negative-transformed image.
    """
    return 255 - image


def is_bubble_filled(contour, eroded_image):
    """
    Checks if a given contour corresponds to a filled bubble.

    Parameters:
        contour (numpy.ndarray): The contour to evaluate.
        eroded_image (numpy.ndarray): The eroded binary image.

    Returns:
        bool: True if the bubble is filled, False otherwise.
    """
    x, y, w, h = cv2.boundingRect(contour)
    choice_region = eroded_image[y:y + h, x:x + w]
    filled_pixels = np.sum(choice_region == 255)
    return filled_pixels >= 10


def extract_answers_region(paper):
    """
    Extracts the region of the paper containing the answers.

    Parameters:
        paper (numpy.ndarray): The scanned image of the paper.

    Returns:
        numpy.ndarray: The region of the paper containing the answers.
    """
    height, _ = paper.shape[:2]
    answer_region_start = height // 3
    return paper[answer_region_start:, :]


def process_row(x_list, eroded_image, bubble_width, student_answers, student_answers_contours,
                student_answers_validations, current_row, answers_per_question):
    """
    Processes a single row of bubbles to determine the answers.

    Parameters:
        x_list (list): List of x-coordinates and contours for the current row.
        eroded_image (numpy.ndarray): Eroded binary image.
        bubble_width (int): Width of a bubble.
        student_answers (numpy.ndarray): Array to store student answers.
        student_answers_contours (list): List to store contours of selected answers.
        student_answers_validations (numpy.ndarray): Array to validate multiple selections.
        current_row (int): The row number being processed.
        answers_per_question (int): Number of answer choices per question.
    """
    x_list.sort(key=lambda pair: pair[0])
    question_index = current_row
    answer_choice = 1

    for i, (x, contour) in enumerate(x_list):
        if i > 0 and x - x_list[i - 1][0] > 2.5 * bubble_width:
            question_index += answers_per_question
            answer_choice = 1

        if is_bubble_filled(contour, eroded_image):
            student_answers[question_index] = answer_choice
            student_answers_contours[question_index] = contour
            student_answers_validations[question_index] += 1

        answer_choice += 1


def get_student_answers(paper, model_answers):
    """
    Extracts and evaluates student answers from a scanned paper against the model answers.

    Parameters:
        paper (numpy.ndarray): The scanned image of the paper.
        model_answers (list[int]): The correct answers for the questions.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Annotated paper image with graded answers.
            - numpy.ndarray: Array of student's answers.
            - numpy.ndarray: Array of grades for each question (1 for correct, 0 for incorrect).
    """
    # Convert to grayscale and apply adaptive thresholding
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 83, 12
    )

    # Apply negative transformation and erosion
    negative_image = negative_transformation(binary_image)
    eroded_image = cv2.erode(negative_image, np.ones((6, 6), np.uint8), iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        negative_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # Filter contours to identify bubbles
    bubble_contours = []
    contour_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

        if (
                0.5 <= aspect_ratio <= 1.5
                and len(approximated_contour) >= 4
                and 30 < cv2.contourArea(contour)
                and cv2.contourArea(contour) > 1.5 * cv2.arcLength(contour, True)
        ):
            bubble_contours.append(contour)
            contour_areas.append(cv2.contourArea(contour))

    # Retain bubbles with areas close to the median area
    median_area = np.median(contour_areas)
    bubble_contours = [
        contour
        for contour, area in zip(bubble_contours, contour_areas)
        if abs(area - median_area) <= 0.1 * median_area
    ]

    # Sort bubbles top-to-bottom
    sorted_contours, _ = imcnts.sort_contours(bubble_contours, method="top-to-bottom")

    # Determine number of questions and answers
    x_prev, y_prev, bubble_width, _ = cv2.boundingRect(sorted_contours[0])
    first_row = [x_prev]

    for contour in sorted_contours[1:]:
        x, y, _, _ = cv2.boundingRect(contour)
        if abs(y - y_prev) > 3:
            break
        first_row.append(x)
        y_prev = y

    first_row.sort()
    bubbles_per_row = len(first_row)
    questions_per_row = sum(
        1 for i in range(1, bubbles_per_row) if first_row[i] - first_row[i - 1] > 2.5 * bubble_width
    ) + 1

    answers_per_question = bubbles_per_row // questions_per_row
    total_questions = len(bubble_contours) // answers_per_question

    # Analyze answers
    student_answers = np.zeros(total_questions, dtype=int)
    student_answers_contours = [None] * total_questions
    student_answers_validations = np.zeros(total_questions, dtype=int)

    current_row = 0
    x_list = [[x_prev, sorted_contours[0]]]
    for contour in sorted_contours[1:]:
        x, y, _, _ = cv2.boundingRect(contour)
        if abs(y - y_prev) > 3:
            # Process previous row
            process_row(
                x_list, eroded_image, bubble_width, student_answers, student_answers_contours,
                student_answers_validations, current_row, answers_per_question
            )
            x_list.clear()
            current_row += 1

        x_list.append([x, contour])
        x_prev = x
        y_prev = y

    # Process the last row
    process_row(
        x_list, eroded_image, bubble_width, student_answers, student_answers_contours,
        student_answers_validations, current_row, answers_per_question
    )

    # Annotate answers and compute grades
    annotated_paper = paper.copy()
    grades = np.zeros(total_questions, dtype=int)

    for i in range(total_questions):
        if student_answers_validations[i] != 1:
            student_answers[i] = 0
            grades[i] = 0
        elif student_answers[i] == model_answers[i]:
            cv2.drawContours(annotated_paper, student_answers_contours[i], -1, (0, 255, 0), 2)
            grades[i] = 1
        elif student_answers[i] != 0:
            cv2.drawContours(annotated_paper, student_answers_contours[i], -1, (0, 0, 255), 2)

    return annotated_paper, student_answers, grades
