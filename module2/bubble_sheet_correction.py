from imutils import contours as imcnts
from .pape_extraction import *
from .utilities import negative_transformation


def is_choice_marked(bubble_contour, eroded_image, count_threshold=10):
    """
    Determines whether a bubble is filled based on the sum of white pixels in its region.

    Args:
        count_threshold (int): threshold for shaded pixel.
        bubble_contour (numpy.ndarray): Contour of the bubble.
        eroded_image (numpy.ndarray): Preprocessed eroded binary image.

    Returns:
        bool: True if the bubble is filled, False otherwise.
    """
    x, y, w, h = cv2.boundingRect(bubble_contour)
    bubble_region = eroded_image[y : y + h, x : x + w]
    white_pixel_count = np.sum(bubble_region == 255)
    print("choice", bubble_region.shape, white_pixel_count)
    return white_pixel_count >= count_threshold


def preprocess_image(paper):
    """
    Preprocesses the input image by converting to grayscale, applying adaptive thresholding,
    and performing a negative transformation.

    Args:
        paper: The input image of the answer sheet.

    Returns:
        eroded_image: The preprocessed image after erosion.
        negative_img: The negative transformed binary image.
    """
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    threshold_binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 83, 12
    )
    negative_img = negative_transformation(threshold_binary_image)
    eroded_image = cv2.erode(negative_img, np.ones((6, 6)), iterations=1)
    return eroded_image, negative_img


def detect_bubble_contours(negative_img):
    """
    Detects bubble contours in the input negative transformed image.

    Args:
        negative_img: The negative transformed binary image.

    Returns:
        circles_contours: Filtered list of contours representing bubbles.
    """
    all_contours, _ = cv2.findContours(
        negative_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    circles_contours = []
    areas_of_contours = []

    for contour in all_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon_value = 0.01 * cv2.arcLength(contour, True)
        circle_contour = cv2.approxPolyDP(contour, epsilon_value, True)

        if (
            0.5 <= aspect_ratio <= 1.5
            and len(circle_contour) >= 4
            and cv2.contourArea(contour) > 30
            and cv2.contourArea(contour) > 1.5 * cv2.arcLength(contour, True)
        ):
            circles_contours.append(contour)
            areas_of_contours.append(cv2.contourArea(contour))

    median_circle_area = np.median(areas_of_contours)
    filtered_contours = [
        contour
        for i, contour in enumerate(circles_contours)
        if abs(areas_of_contours[i] - median_circle_area) <= median_circle_area * 0.1
    ]

    return filtered_contours


def process_bubble_answers(sorted_contours, eroded_image, number_of_questions):
    """
    Processes bubble answers to extract the student's responses.

    Args:
        sorted_contours: Contours sorted from top to bottom.
        eroded_image: The preprocessed eroded image.
        number_of_questions: The total number of questions.

    Returns:
        student_answers: Array of student's answers.
        student_answers_contours: Contours of marked bubbles.
        student_answers_validate: Array indicating the validation status of answers.
    """
    student_answers = np.zeros(number_of_questions, dtype=int)
    student_answers_contours = [None] * number_of_questions
    student_answers_validate = np.zeros(number_of_questions, dtype=int)

    (_, _, circle_width, _) = cv2.boundingRect(sorted_contours[0])
    (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])
    curr_row = 0
    x_list = [[x_prev, sorted_contours[0]]]

    for contour in sorted_contours[1:]:
        (x, y, _, _) = cv2.boundingRect(contour)
        if abs(y - y_prev) > 3:
            curr_row = process_row_answers(
                x_list,
                eroded_image,
                student_answers,
                student_answers_contours,
                student_answers_validate,
                curr_row,
                circle_width,
            )
            x_list.clear()
        x_list.append([x, contour])
        y_prev = y

    process_row_answers(
        x_list,
        eroded_image,
        student_answers,
        student_answers_contours,
        student_answers_validate,
        curr_row,
        circle_width,
    )

    return student_answers, student_answers_contours, student_answers_validate


def process_row_answers(
    x_list,
    eroded_image,
    student_answers,
    student_answers_contours,
    student_answers_validate,
    curr_row,
    circle_width,
):
    """
    Processes the answers in a single row of bubbles.

    Args:
        x_list: List of x-coordinates and contours for the current row.
        eroded_image: The preprocessed eroded image.
        student_answers: Array of student's answers.
        student_answers_contours: Contours of marked bubbles.
        student_answers_validate: Array indicating the validation status of answers.
        curr_row: Current row number.
        circle_width: Width of a single bubble.

    Returns:
        Updated row number.
    """
    x_list.sort(key=lambda pair: pair[0])
    question_per_row = 1
    answer = 1
    question_num = curr_row
    for i in range(len(x_list)):
        if (i - 1 >= 0) & ((x_list[i][0] - x_list[i - 1][0]) > (2.5 * circle_width)):
            question_num = curr_row + 15 * question_per_row
            question_per_row += 1
            answer = 1
        if is_choice_marked(x_list[i][1], eroded_image) == 1:
            student_answers[question_num] = answer
            student_answers_contours[question_num] = x_list[i][1]
            student_answers_validate[question_num] += 1
        answer += 1
    return curr_row + 1


def compare_answers_with_model(
    student_answers, student_answers_contours, model_answer, paper
):
    """
    Compares student's answers with the model answers and generates grades.

    Args:
        student_answers: Array of student's answers.
        student_answers_contours: Contours of marked bubbles.
        model_answer: List of correct answers.
        paper: The original answer sheet image.

    Returns:
        output_paper: Image with marked contours indicating correct and incorrect answers.
        grades: Array of grades (1 for correct, 0 for incorrect).
    """
    output_paper = paper.copy()
    grades = np.zeros(len(model_answer), dtype=int)

    for i in range(len(model_answer)):
        if student_answers[i] == model_answer[i]:
            cv2.drawContours(
                output_paper, student_answers_contours[i], -1, (0, 255, 0), 2
            )
            grades[i] = 1
        elif student_answers[i] != 0:
            cv2.drawContours(
                output_paper, student_answers_contours[i], -1, (0, 0, 255), 2
            )

    return output_paper, grades


def get_student_answers(paper, model_answer):
    """
    Main function to process the answer sheet and grade the student's answers.

    Args:
        paper: The input image of the answer sheet.
        model_answer: A list of correct answers for comparison.

    Returns:
        output_paper: The image with contours drawn around the student's marked answers.
        student_answers: An array of student's answers.
        grades: An array of grades (1 for correct, 0 for incorrect).
    """
    eroded_image, negative_img = preprocess_image(paper)
    circles_contours = detect_bubble_contours(negative_img)
    sorted_contours, _ = imcnts.sort_contours(circles_contours, method="top-to-bottom")

    # Copy the input paper for later visualization
    contoured_paper = paper.copy()
    cv2.drawContours(contoured_paper, circles_contours, -1, (0, 0, 255), 2)

    print("shape", contoured_paper.shape)
    plt.imshow(contoured_paper, cmap="gray")
    plt.show()

    number_of_questions = len(model_answer)

    student_answers, student_answers_contours, student_answers_validate = (
        process_bubble_answers(sorted_contours, eroded_image, number_of_questions)
    )

    output_paper, grades = compare_answers_with_model(
        student_answers, student_answers_contours, model_answer, paper
    )
    return output_paper, student_answers, grades
