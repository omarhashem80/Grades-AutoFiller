import matplotlib.pyplot as plt

from bubble_sheet_correction import *
from train_digits import *


def segment_id(code):
    """
    Segments individual digits from the given code image.

    Parameters:
        code (numpy.ndarray): The input image containing the code.

    Returns:
        list: A list of segmented digit images.
    """
    contours, _ = cv2.findContours(code, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate bounding rectangles for the contours
    bounding_rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_rectangles = [rect for rect in bounding_rectangles if rect[2] * rect[3] > 100]
    bounding_rectangles = sorted(bounding_rectangles, key=lambda x: x[0])
    digits = []

    # Extract each digit using the bounding rectangle
    for rect in bounding_rectangles:
        x, y, w, h = rect
        digit = code[y:y + h, x:x + w]
        digits.append(digit)

    return digits


def extract_student_code(paper):
    """
    Extracts the student code area from the scanned paper image.

    Parameters:
        paper (numpy.ndarray): The scanned image of the paper.

    Returns:
        numpy.ndarray: The processed image of the extracted student code.
    """
    # Convert to grayscale and apply adaptive thresholding
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    threshold_binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 15
    )
    contours, _ = cv2.findContours(threshold_binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        target_area = 30000
        for contour in contours:
            epsilon_value = 0.01 * cv2.arcLength(contour, True)
            paper_contour = cv2.approxPolyDP(contour, epsilon_value, True)

            # Look for the largest 4-sided contour
            if len(paper_contour) == 4 and cv2.contourArea(contour) > 0.2 * target_area:
                code = image_transform(paper, paper_contour.reshape(4, 2))

    # Further process the extracted code
    gray_code = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
    blurred_code = cv2.GaussianBlur(gray_code, (5, 5), 0.5)
    threshold_binary_code = cv2.adaptiveThreshold(
        blurred_code, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10
    )
    negative_code = negative_transformation(threshold_binary_code)
    eroded_image = cv2.erode(negative_code, np.ones((2, 2), np.uint8), iterations=1)
    dilated_image = cv2.dilate(eroded_image, np.ones((3, 3), np.uint8), iterations=1)
    return dilated_image


def crop_code(code):
    """
    Crops the relevant area containing the code from the given image.

    Parameters:
        code (numpy.ndarray): The binary image containing the code.

    Returns:
        numpy.ndarray: The cropped image containing the relevant code area.
    """
    # Dilate the image to emphasize the region of interest
    kernel = np.ones((10, 25), np.uint8)
    dilated_image = cv2.dilate(code, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = code[y:y + h, x:x + w]
    return cropped_image


def preprocess_bubble_sheet(paper):
    """
    Preprocesses the bubble sheet image by converting it to grayscale, applying thresholding,
    negative transformation, and erosion.

    Args:
        paper (numpy.ndarray): Image of the bubble sheet.

    Returns:
        tuple: Eroded image, negative image, and the thresholded binary image.
    """
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    threshold_binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 15
    )
    negative_img = negative_transformation(threshold_binary_image)
    eroded_image = cv2.erode(negative_img, np.ones((7, 7)), iterations=1)
    return eroded_image, negative_img, threshold_binary_image


def filter_bubble_contours(negative_img):
    """
    Filters contours to identify valid bubbles based on their shape and area.

    Args:
        negative_img (numpy.ndarray): Negative transformed binary image.

    Returns:
        list: Filtered list of contours representing bubbles.
    """
    all_contours, _ = cv2.findContours(negative_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        circles_contours[i] for i, area in enumerate(areas_of_contours)
        if abs(area - median_circle_area) <= median_circle_area * 0.1
    ]
    return filtered_contours


def calculate_row_and_columns(circles_contours, circle_width):
    """
    Calculates the number of questions per row and answers per question based on the first row of bubbles.

    Args:
        circles_contours (list): List of bubble contours.
        circle_width (int): Width of a single bubble.

    Returns:
        tuple: Number of questions per row and answers per question.
    """
    # Analyze the first row of bubbles
    (x_prev, y_prev, _, _) = cv2.boundingRect(circles_contours[0])
    first_row = [x_prev]
    for contour in circles_contours[1:]:
        (x, y, _, _) = cv2.boundingRect(contour)
        if abs(y - y_prev) > 3:
            break
        first_row.append(x)
    first_row.sort()

    questions_number_per_row = 1
    for i in range(1, len(first_row)):
        if first_row[i] - first_row[i - 1] > 2.5 * circle_width:
            questions_number_per_row += 1
    answers_number_per_question = len(first_row) // questions_number_per_row
    return questions_number_per_row, answers_number_per_question


def detect_marked_answers(sorted_contours, eroded_image, number_of_questions, circle_width):
    """
    Detects marked answers for each question based on bubble selections.

    Args:
        sorted_contours (list): Sorted list of contours.
        eroded_image (numpy.ndarray): Eroded image for analyzing marked bubbles.
        number_of_questions (int): Total number of questions.
        circle_width (int): Width of a single bubble.

    Returns:
        tuple: Student answers array and chosen contours.
    """
    student_answers = np.zeros(number_of_questions, dtype=int)
    chosen_contours = [None] * number_of_questions
    student_answers_validate = np.zeros(number_of_questions, dtype=int)

    curr_row = 0
    x_list = []
    (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])

    for contour in sorted_contours:
        (x, y, _, _) = cv2.boundingRect(contour)
        if abs(y - y_prev) > 3:
            process_row(x_list, eroded_image, student_answers, chosen_contours, student_answers_validate, curr_row,
                        circle_width)
            x_list.clear()
            curr_row += 1

        x_list.append((x, contour))
        y_prev = y

    process_row(x_list, eroded_image, student_answers, chosen_contours, student_answers_validate, curr_row,
                circle_width)

    for i, validate_count in enumerate(student_answers_validate):
        if validate_count != 1:
            student_answers[i] = -1
        else:
            student_answers[i] -= 1

    return student_answers, chosen_contours


def process_row(x_list, eroded_image, student_answers, chosen_contours, student_answers_validate, curr_row,
                circle_width):
    """
    Processes a single row of bubbles to determine marked answers.

    Args:
        x_list (list): List of x-coordinates and contours for the current row.
        eroded_image (numpy.ndarray): Eroded image for analyzing marked bubbles.
        student_answers (numpy.ndarray): Array to store student answers.
        chosen_contours (list): List of chosen contours for each question.
        student_answers_validate (numpy.ndarray): Validation array for answers.
        curr_row (int): Current row number.
        circle_width (int): Width of a single bubble.
    """
    x_list.sort(key=lambda pair: pair[0])
    question_per_row = 1
    answer = 1
    question_num = curr_row

    for i in range(len(x_list)):
        if i > 0 and (x_list[i][0] - x_list[i - 1][0]) > 2.5 * circle_width:
            question_num = curr_row + question_per_row
            question_per_row += 1
            answer = 1
        if is_choice_marked(x_list[i][1], eroded_image) == 1:
            student_answers[question_num] = answer
            chosen_contours[question_num] = x_list[i][1]
            student_answers_validate[question_num] += 1
        answer += 1


def get_student_bubble_code(paper):
    """
    Main function to process the scanned image of a bubble sheet and extract student responses.

    Args:
        paper (numpy.ndarray): Image of the bubble sheet.

    Returns:
        tuple: Contoured paper image and array of student answers.
    """
    eroded_image, negative_img, _ = preprocess_bubble_sheet(paper)

    plt.imshow(eroded_image, cmap='gray')
    plt.show()
    circles_contours = filter_bubble_contours(negative_img)
    contoured_paper = paper.copy()

    sorted_contours, _ = imcnts.sort_contours(circles_contours, method='top-to-bottom')

    # Copy the input paper for later visualization
    contoured_paper = paper.copy()
    cv2.drawContours(contoured_paper, circles_contours, -1, (0, 0, 255), 2)

    (_, __, circle_width, ___) = cv2.boundingRect(circles_contours[0])

    questions_per_row, answers_per_question = calculate_row_and_columns(circles_contours, circle_width)
    number_of_questions = len(circles_contours) // answers_per_question

    student_answers, chosen_contours = detect_marked_answers(sorted_contours, eroded_image, number_of_questions,
                                                             circle_width)
    cv2.drawContours(contoured_paper, chosen_contours, -1, (255, 0, 0), 2)

    return contoured_paper, student_answers


def get_code_prediction(digits):
    """
    Predicts the numeric codes from a list of digit images.

    Args:
        digits (list of numpy.ndarray): A list of images representing digit regions.

    Returns:
        list: A list of integers representing the predicted digits for each input image.
    """
    arr = []  # List to store predictions for each digit
    for i, image in enumerate(digits):
        # Apply erosion to the image to enhance the features
        eroded_image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)

        # Resize the image to 28x28 pixels, the input size expected by the digit recognition model
        resized_image = cv2.resize(eroded_image, (28, 28))

        # Predict the digit from the processed image
        arr.append(predict_digit(resized_image))

    return arr
