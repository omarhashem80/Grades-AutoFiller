import os
import cv2
import numpy as np
import joblib
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import gradio as gr
from Module1.Grade_sheet_processor import GradeSheetOCR
from module2.main import (
    fill_grades_in_sheet,
    read_answers_from_file,
    process_image_and_grades,
)
from module2.pape_extraction import extract_paper
from module2.bubble_sheet_correction import get_student_answers
from module2.code_extraction import (
    extract_student_code,
    crop_code,
    get_student_bubble_code,
    segment_id,
    get_code_prediction,
)
from module2.utilities import extract_answers_region, extract_bubble_code


class GradeSheetOCRApp:
    def __init__(self):
        self.ocr = GradeSheetOCR()

    def create_gradio_interface(self):
        def process_grade_sheet(image, code_ocr, number_ocr):
            try:
                temp_image_path = "temp_uploaded_image.jpg"
                image.save(temp_image_path)
                codes, arabic_names, english_names, numbers = (
                    self.ocr.process_grade_sheet(
                        temp_image_path,
                        code_ocr_method="ocr" if code_ocr else "training",
                        number_ocr_method="ocr" if number_ocr else "training",
                    )
                )
                excel_path = self.ocr.create_excel_file(
                    codes,
                    arabic_names,
                    english_names,
                    numbers,
                )
                return f"Excel file created: {excel_path}"
            except Exception as e:
                return f"Error processing grade sheet: {str(e)}"

        interface = gr.Interface(
            fn=process_grade_sheet,
            inputs=[
                gr.Image(type="pil", label="Upload Grade Sheet Image"),
                gr.Checkbox(label="Use Code OCR"),
                gr.Checkbox(label="Use Number OCR"),
            ],
            outputs=gr.Textbox(label="Result"),
            title="Grade Sheet OCR",
            description="Upload a grade sheet image to extract student information",
        )
        return interface


class BubbleSheetCorrectionApp:
    def create_gradio_interface(self):
        def process_bubble_sheet(image, model_answer_file):
            try:
                temp_image_path = "temp_bubble_sheet.jpg"
                image.save(temp_image_path)
                model_answer_path = model_answer_file.name
                img = cv2.imread(temp_image_path)
                print(img.shape)
                process_image_and_grades(img, model_answer_path)
                return "Bubble sheet processed successfully. Check outputs at the outputs folder."
            except Exception as e:
                return f"Error processing bubble sheet: {str(e)}"

        interface = gr.Interface(
            fn=process_bubble_sheet,
            inputs=[
                gr.Image(type="pil", label="Upload Bubble Sheet Image"),
                gr.File(type="filepath", label="Upload Model Answer Text File"),
            ],
            outputs=gr.Textbox(label="Result"),
            title="Bubble Sheet Correction",
            description="Upload a bubble sheet image and model answer text file to process",
        )
        return interface


def main():
    grade_sheet_app = GradeSheetOCRApp()
    bubble_sheet_app = BubbleSheetCorrectionApp()

    grade_sheet_interface = grade_sheet_app.create_gradio_interface()
    bubble_sheet_interface = bubble_sheet_app.create_gradio_interface()

    with gr.Blocks() as demo:
        with gr.Tab("Grade Sheet OCR"):
            grade_sheet_interface.render()
        with gr.Tab("Bubble Sheet Correction"):
            bubble_sheet_interface.render()

    demo.launch()


if __name__ == "__main__":
    main()
