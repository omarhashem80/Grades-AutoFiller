import cv2
import numpy as np
import os
import pandas as pd
from pape_extraction import *
from bubble_sheet_correction import *
from train_digits import *
from code_extraction import *


def fill_grades_in_sheet(student_code, student_grades):
    """
    Fills the grades of a student into an Excel sheet.

    If the Excel file exists, it appends the new student grades; otherwise, it creates a new file.

    Parameters:
    - student_code: A string representing the student's code.
    - student_grades: A list of grades (answers) for each question.
    """
    main_relative_path = ""
    file_name = main_relative_path + 'output.xlsx'

    # Check if the file exists
    if os.path.isfile(file_name):
        existing_data = pd.read_excel(file_name)

        new_data = {'Code': [student_code]}
        new_data.update({f'Q{i + 1}': [answer] for i, answer in enumerate(student_grades)})
        new_df = pd.DataFrame(new_data)

        combined_df = pd.concat([existing_data, new_df], ignore_index=True)
        combined_df.to_excel(file_name, index=False)
        print(f"Excel sheet '{file_name}' updated successfully.")
    else:
        data = {'Code': [student_code]}
        data.update({f'Q{i + 1}': [answer] for i, answer in enumerate(student_grades)})
        df = pd.DataFrame(data)

        df.to_excel(file_name, index=False)
        print(f"Excel sheet '{file_name}' created successfully.")


def read_answers_from_file(file_path):
    """
    Reads answers from a file where each line contains an answer.

    Parameters:
    - file_path: Path to the file containing the answers.

    Returns:
    - A list of integer answers.
    """
    with open(file_path, 'r') as file:
        answers = [int(line.strip()) for line in file]

    return answers


def process_image_and_grades(img, file):
    """
    Processes the image and grades by extracting the paper, bubble code, written code, and answers.

    The results are saved in the student's folder, and the grades are recorded in the Excel sheet.

    Parameters:
    - img: The input image of the student’s paper.
    - file: Path to the file containing the answers.
    """
    model_answer = read_answers_from_file(file)

    # Resize image for processing
    img = cv2.resize(img, (800, 1000))
    paper = extract_paper_region(img)

    # Get student bubble code (Method 1)
    bubble_code = extract_bubble_code(paper)
    student_bubble_code_img, student_bubble_code = get_student_bubble_code(bubble_code)

    # Get student written code (Method 2)
    code = extract_student_code(paper)
    cropped_code = crop_code(code)
    digits = segment_id(cropped_code)
    written_code = get_code_prediction(digits)
    written_code_str = ''.join(written_code)

    # Get student answers
    answers_region = extract_answers_region(paper)
    print(answers_region, model_answer)
    answers_img, answers, grades = get_student_answers(answers_region, model_answer)

    output_folder = f'outputs/{written_code_str}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save images in the student's folder
    cv2.imwrite(os.path.join(output_folder, 'student_bubble_code.jpg'), student_bubble_code_img)
    cv2.imwrite(os.path.join(output_folder, 'student_written_code.jpg'), cropped_code)
    cv2.imwrite(os.path.join(output_folder, 'student_answers.jpg'), answers_img)

    print("Student Bubble Code: ", student_bubble_code)
    print("Student Written Code: ", written_code)
    print("Student Answers: ", answers)
    print("Student Grades: ", grades, "\n")

    # Record the grades in the Excel sheet
    fill_grades_in_sheet(written_code_str, grades)