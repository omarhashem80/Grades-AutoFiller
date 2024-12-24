from imutils import contours as imcnts
from pape_extraction import *


def negative_transformation(image):
    """
    Applies negative transformation to the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Negative-transformed image.
    """
    return 255 - image


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
    bubble_region = eroded_image[y:y + h, x:x + w]
    white_pixel_count = np.sum(bubble_region == 255)
    print('choice', bubble_region.shape, white_pixel_count)
    return white_pixel_count >= count_threshold


def extract_answers_region(paper_image):
    """
    Extracts the region of interest containing the answers from the paper.

    Args:
        paper_image (numpy.ndarray): Input scanned paper image.

    Returns:
        numpy.ndarray: Cropped region containing the answers.
    """
    height, _ = paper_image.shape[:2]
    cropped_height = height // 3
    return paper_image[cropped_height:, :]


def get_student_answers(paper, model_answer):
    """
    This function processes an image of a student's answer sheet, detects the marked choices in the bubbles,
    compares them to the model answers, and returns the marked answer sheet, the student's answers, and grades.

    Args:
        paper: The input image of the answer sheet.
        model_answer: A list of correct answers for comparison.

    Returns:
        output_paper: The image with contours drawn around the student's marked answers.
        student_answers: An array of student's answers (0 for no answer, 1 for marked answer).
        grades: An array of grades (1 for correct, 0 for incorrect).
    """
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    threshold_binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 83,
                                                   12)

    # Apply a negative transformation to the thresholded image
    negative_img = negative_transformation(threshold_binary_image)

    # print('shape', negative_img.shape)
    # plt.imshow(negative_img, cmap='gray')
    # plt.show()

    # Erode the image to reduce noise
    eroded_image = cv2.erode(negative_img, np.ones((6, 6)), iterations=1)

    # print('shape', eroded_image.shape)
    # plt.imshow(eroded_image, cmap='gray')
    # plt.show()

    # Find all external contours in the negative image
    all_contours, _ = cv2.findContours(negative_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize lists to store valid circle contours and their areas
    circles_contours = []
    areas_of_contours = []

    # Loop through each contour and check if it meets the criteria for being a bubble
    for contour in all_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon_value = 0.01 * cv2.arcLength(contour, True)
        circle_contour = cv2.approxPolyDP(contour, epsilon_value, True)

        # If the contour is within certain area and aspect ratio limits, consider it a valid bubble
        if (0.5 <= aspect_ratio <= 1.5 and len(circle_contour) >= 4 and cv2.contourArea(
                contour) > 30 and cv2.contourArea(contour) > 1.5 * cv2.arcLength(contour, True)):
            circles_contours.append(contour)
            areas_of_contours.append(cv2.contourArea(contour))

    # Calculate the median area of all detected contours
    median_circle_area = np.median(areas_of_contours)

    # Filter out contours that have an area far from the median
    circles_contours_temp = []
    for i, area in enumerate(areas_of_contours):
        if abs(area - median_circle_area) <= median_circle_area * 0.1:
            circles_contours_temp.append(circles_contours[i])

    circles_contours = circles_contours_temp

    # Copy the input paper for later visualization
    contoured_paper = paper.copy()
    cv2.drawContours(contoured_paper, circles_contours, -1, (0, 0, 255), 2)

    print('shape', contoured_paper.shape)
    plt.imshow(contoured_paper, cmap='gray')
    plt.show()

    # Sort the contours from top to bottom to process them row by row
    sorted_contours, _ = imcnts.sort_contours(circles_contours, method='top-to-bottom')

    # Get the width of the first circle (bubble) to estimate spacing
    (_, __, circleWidth, ___) = cv2.boundingRect(circles_contours[0])

    # Get the bounding box of the first contour
    (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])

    # Initialize the first row with the x-coordinate of the first contour
    first_row = [x_prev]
    for contour in sorted_contours[1:]:
        (x, y, _, _) = cv2.boundingRect(contour)

        # Break when we reach a new row (vertical distance between contours is large)
        if abs(y - y_prev) > 3:
            break
        first_row.append(x)

    # Sort the first row by the x-coordinate
    first_row.sort()

    number_of_questions = len(model_answer)

    # Initialize arrays to store student answers, contours, and validation results
    student_answers = np.zeros(number_of_questions, dtype=int)
    student_answers_contours = [None] * number_of_questions
    student_answers_validate = np.zeros(number_of_questions, dtype=int)

    # Initialize variables to track the current row and contours
    (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])
    curr_row = 0
    x_list = [[x_prev, sorted_contours[0]]]

    # Loop through all sorted contours to identify the marked answers
    for contour in sorted_contours[1:]:
        (x, y, _, _) = cv2.boundingRect(contour)

        # Check for row changes and process answers in each row
        if abs(y - y_prev) > 3:
            x_list.sort(key=lambda pair: pair[0])  # Sort by x-coordinate

            # Process each x-coordinate and mark answers
            question_per_row = 1
            answer = 1
            question_num = curr_row
            for i in range(len(x_list)):
                if (i - 1 >= 0) & ((x_list[i][0] - x_list[i - 1][0]) > (2.5 * circleWidth)):
                    question_num = curr_row + 15 * question_per_row
                    question_per_row += 1
                    answer = 1
                if is_choice_marked(x_list[i][1], eroded_image) == 1:
                    student_answers[question_num] = answer
                    student_answers_contours[question_num] = x_list[i][1]
                    student_answers_validate[question_num] += 1
                answer += 1
            x_list.clear()
            curr_row += 1

        # Add current contour to the x_list for processing
        x_list.append([x, contour])
        y_prev = y

    # Final sorting and marking of answers for the last row
    x_list.sort(key=lambda pair: pair[0])
    question_per_row = 1
    answer = 1
    question_num = curr_row
    for i in range(len(x_list)):
        if (i - 1 >= 0) & ((x_list[i][0] - x_list[i - 1][0]) > (2.5 * circleWidth)):
            question_num = curr_row + 15 * question_per_row
            question_per_row += 1
            answer = 1
        if is_choice_marked(x_list[i][1], eroded_image) == 1:
            student_answers[question_num] = answer
            student_answers_contours[question_num] = x_list[i][1]
            student_answers_validate[question_num] += 1
        answer += 1
    x_list.clear()

    # Create a copy of the input paper to visualize the results
    output_paper = paper.copy()
    grades = np.zeros(number_of_questions, dtype=int)

    # Compare student's answers with the model answers and mark them
    for i in range(len(model_answer)):
        if student_answers_validate[i] != 1:
            student_answers[i] = 0
            grades[i] = 0
        if student_answers[i] == model_answer[i]:
            cv2.drawContours(output_paper, student_answers_contours[i], -1, (0, 255, 0), 2)
            grades[i] = 1
        elif student_answers[i] != 0:
            cv2.drawContours(output_paper, student_answers_contours[i], -1, (0, 0, 255), 2)
            grades[i] = 0

    return output_paper, student_answers, grades
