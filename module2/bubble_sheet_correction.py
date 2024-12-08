from imutils import contours as imcnts
from paper_extraction import *
import cv2
import numpy as np


def negative_transformation(image):
    """
    Applies a negative transformation to an image.

    Args:
        image (numpy.ndarray): Input binary or grayscale image.

    Returns:
        numpy.ndarray: Negative transformed image, where pixel values are inverted (255 - original value).
    """
    negative_image = 255 - image
    return negative_image


def get_choice(contour, eroded_image):
    """
    Determines if a specific contour represents a filled bubble (student's answer).

    Args:
        contour (numpy.ndarray): The contour to evaluate.
        eroded_image (numpy.ndarray): The eroded binary image containing the bubbles.

    Returns:
        bool: True if the bubble is filled, otherwise False.
    """
    x, y, w, h = cv2.boundingRect(contour)
    choice_region = eroded_image[y:y + h, x:x + w]
    total = np.sum(choice_region == 255)
    return total >= 10  # Threshold for detecting filled bubbles


def extract_answers_region(paper):
    """
    Crops the answer section from the scanned paper image.

    Args:
        paper (numpy.ndarray): The full scanned paper image.

    Returns:
        numpy.ndarray: The cropped region containing the answer bubbles.
    """
    x, y = paper.shape[:2]
    new_x = (x // 3)
    segment = paper[new_x:, :]
    return segment


def get_student_answers(paper, model_answer):
    """
    Extracts and evaluates student answers from the answer sheet.

    Args:
        paper (numpy.ndarray): The scanned paper image containing the answers.
        model_answer (numpy.ndarray): The correct answers for grading.

    Returns:
        tuple:
            - output_paper (numpy.ndarray): Annotated paper image showing correct and incorrect answers.
            - student_answers (numpy.ndarray): Detected answers by the student.
            - grades (numpy.ndarray): Grades for each question (1 for correct, 0 for incorrect).
    """
    # Step 1: Convert to grayscale and apply adaptive thresholding
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 83, 12)

    # Step 2: Apply negative transformation and morphological erosion
    negative_image = negative_transformation(binary_image)
    eroded_image = cv2.erode(negative_image, np.ones((6, 6)), iterations=1)

    # Step 3: Detect external contours
    all_contours, _ = cv2.findContours(negative_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    circles_contours = []
    areas_of_contours = []

    # Step 4: Filter contours based on aspect ratio, area, and shape
    for contour in all_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon = 0.01 * cv2.arcLength(contour, True)
        circle_contour = cv2.approxPolyDP(contour, epsilon, True)

        if (0.5 <= aspect_ratio <= 1.5 and len(circle_contour) >= 4 and
                cv2.contourArea(contour) > 30 and
                cv2.contourArea(contour) > 1.5 * cv2.arcLength(contour, True)):
            circles_contours.append(contour)
            areas_of_contours.append(cv2.contourArea(contour))

    # Step 5: Filter bubbles based on area consistency
    median_circle_area = np.median(areas_of_contours)
    circles_contours_temp = [
        circles_contours[i] for i, area in enumerate(areas_of_contours)
        if abs(area - median_circle_area) <= median_circle_area * 0.1
    ]
    circles_contours = circles_contours_temp

    # Step 6: Sort contours top-to-bottom for processing
    sorted_contours, _ = imcnts.sort_contours(circles_contours, method='top-to-bottom')
    (_, __, circle_width, ___) = cv2.boundingRect(circles_contours[0])

    # Step 7: Determine number of questions and answers per question
    first_row = [cv2.boundingRect(contour)[0] for contour in sorted_contours[:15]]
    first_row.sort()
    questions_number_per_row = 1
    for i in range(1, len(first_row)):
        if first_row[i] - first_row[i - 1] > 2.5 * circle_width:
            questions_number_per_row += 1
    answers_number_per_question = len(first_row) // questions_number_per_row
    number_of_questions = len(circles_contours) // answers_number_per_question

    # Step 8: Initialize arrays for student answers and validation
    student_answers = np.zeros(number_of_questions, dtype=int)
    student_answers_contours = [None] * number_of_questions
    student_answers_validate = np.zeros(number_of_questions, dtype=int)

    # Step 9: Assign answers row by row
    curr_row = 0
    x_list = [[cv2.boundingRect(sorted_contours[0])[0], sorted_contours[0]]]
    for contour in sorted_contours[1:]:
        x, y, _, _ = cv2.boundingRect(contour)
        if abs(y - cv2.boundingRect(x_list[-1][1])[1]) > 3:
            x_list.sort(key=lambda pair: pair[0])
            for i, (x_val, cont) in enumerate(x_list):
                question_num = curr_row + 15 * (i // answers_number_per_question)
                if get_choice(cont, eroded_image):
                    student_answers[question_num] = i % answers_number_per_question + 1
                    student_answers_contours[question_num] = cont
                    student_answers_validate[question_num] += 1
            x_list.clear()
            curr_row += 1
        x_list.append([x, contour])

    # Step 10: Grade answers and annotate the paper
    output_paper = paper.copy()
    grades = np.zeros(number_of_questions, dtype=int)
    for i in range(len(student_answers_validate)):
        if student_answers_validate[i] != 1:
            student_answers[i] = 0
        if student_answers[i] == model_answer[i]:
            cv2.drawContours(output_paper, student_answers_contours[i], -1, (0, 255, 0), 2)
            grades[i] = 1
        elif student_answers[i] != 0:
            cv2.drawContours(output_paper, student_answers_contours[i], -1, (0, 0, 255), 2)

    return output_paper, student_answers, grades
