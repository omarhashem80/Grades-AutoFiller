from bubble_sheet_correction import *
from train_digits import *


def segment_id(code):
    contours, _ = cv2.findContours(code, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_rectangles = [rect for rect in bounding_rectangles if rect[2] * rect[3] > 100]
    bounding_rectangles = sorted(bounding_rectangles, key=lambda x: x[0])
    digits = []

    for rect in bounding_rectangles:
        x, y, w, h = rect
        digit = code[y:y + h, x:x + w]
        digits.append(digit)

    return digits


def extract_bubble_code(paper):
    x, y = paper.shape[:2]

    new_x = (x // 3) + 10
    new_y = (y // 2) + 40

    segment = paper[:new_x, :new_y]

    return segment


def extract_student_code(paper):
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    threshold_binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155,
                                                   10)
    contours, _ = cv2.findContours(threshold_binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
    threshold_binary_code = cv2.adaptiveThreshold(blurred_code, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155,
                                                  10)
    negative_code = negative_transformation(threshold_binary_code)
    eroded_image = cv2.erode(negative_code, np.ones((2, 2), np.uint8), iterations=1)
    dilated_image = cv2.dilate(eroded_image, np.ones((3, 3), np.uint8), iterations=1)
    return dilated_image


def crop_code(code):
    kernel = np.ones((10, 25), np.uint8)
    dilated_image = cv2.dilate(code, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = code[y:y + h, x:x + w]
    return cropped_image


def get_student_bubble_code(paper):
    gray_image = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    threshold_binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255,
                                                   15)
    negative_img = negative_transformation(threshold_binary_image)
    eroded_image = cv2.erode(negative_img, np.ones((7, 7)), iterations=1)
    all_contours, _ = cv2.findContours(negative_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    circles_contours = []
    areas_of_contours = []
    for contour in all_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / h
        epsilon_value = 0.01 * cv2.arcLength(contour, True)
        circle_contour = cv2.approxPolyDP(contour, epsilon_value, True)
        if (0.5 <= aspect_ratio <= 1.5 and len(circle_contour) >= 4 and cv2.contourArea(
                contour) > 30 and cv2.contourArea(contour) > 1.5 * cv2.arcLength(contour, True)):
            circles_contours.append(contour)
            areas_of_contours.append(cv2.contourArea(contour))

    # Get Only Bubbles
    median_circle_area = np.median(areas_of_contours)
    circles_contours_temp = []
    for i, area in enumerate(areas_of_contours):
        if abs(area - median_circle_area) <= median_circle_area * 0.1:
            circles_contours_temp.append(circles_contours[i])

    circles_contours = circles_contours_temp

    contoured_paper = paper.copy()

    sorted_contours, _ = imcnts.sort_contours(circles_contours, method='top-to-bottom')
    (_, __, circleWidth, ___) = cv2.boundingRect(circles_contours[0])
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
        if first_row[i] - first_row[i - 1] > 2.5 * circleWidth:
            questions_number_per_row += 1

    answers_number_per_question = circles_number // questions_number_per_row
    number_of_questions = len(circles_contours) // answers_number_per_question

    student_answers = np.zeros(number_of_questions, dtype=int)
    chosen_contours = [None] * number_of_questions
    student_answers_validate = np.zeros(number_of_questions, dtype=int)
    (x_prev, y_prev, _, _) = cv2.boundingRect(sorted_contours[0])
    curr_row = 0
    x_list = [[x_prev, sorted_contours[0]]]
    for contour in sorted_contours[1:]:
        (x, y, _, _) = cv2.boundingRect(contour)

        if abs(y - y_prev) > 3:
            x_list.sort(key=lambda pair: pair[0])
            question_per_row = 1
            answer = 1
            question_num = curr_row
            for i in range(len(x_list)):
                if (i - 1 >= 0) & ((x_list[i][0] - x_list[i - 1][0]) > (2.5 * circleWidth)):
                    question_num = curr_row + question_per_row
                    question_per_row += 1
                    answer = 1
                if is_choice_marked(x_list[i][1], eroded_image) == 1:
                    student_answers[question_num] = answer
                    chosen_contours[question_num] = x_list[i][1]
                    student_answers_validate[question_num] += 1
                answer += 1
            x_list.clear()
            curr_row += 1

        x_list.append([x, contour])
        # x_prev = x
        y_prev = y

    x_list.sort(key=lambda pair: pair[0])
    question_per_row = 1
    answer = 1
    question_num = curr_row
    for i in range(len(x_list)):
        if (i - 1 >= 0) & ((x_list[i][0] - x_list[i - 1][0]) > (2.5 * circleWidth)):
            question_num = curr_row + question_per_row
            question_per_row += 1
            answer = 1
        if is_choice_marked(x_list[i][1], eroded_image) == 1:
            student_answers[question_num] = answer
            chosen_contours[question_num] = x_list[i][1]
            student_answers_validate[question_num] += 1
        answer += 1
    x_list.clear()

    for i in range(len(student_answers_validate)):
        if student_answers_validate[i] != 1:
            student_answers[i] = -1
        else:
            student_answers[i] -= 1

    cv2.drawContours(contoured_paper, chosen_contours, -1, (255, 0, 0), 2)
    return contoured_paper, student_answers


def get_code_prediction(digits):
    arr = []
    for i, image in enumerate(digits):
        eroded_image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)
        resized_image = cv2.resize(eroded_image, (28, 28))
        arr.append(predict_digit(resized_image))
    return arr

