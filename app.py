import os
import cv2
import numpy as np
import joblib
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import gradio as gr
from Module1.Grade_sheet_processor import GradeSheetOCR
from module2.main import fill_grades_in_sheet, read_answers_from_file
from module2.pape_extraction import extract_paper_region
from module2.bubble_sheet_correction import extract_answers_region, get_student_answers
from module2.code_extraction import (
    extract_bubble_code,
    extract_student_code,
    crop_code,
    get_student_bubble_code,
    segment_id,
    get_code_prediction,
)


class GradiosApp:
    """
    A Gradio-based application to process grade sheets and bubble sheets.
    """

    def __init__(self):
        self.ocr = GradeSheetOCR()

    def process_grade_sheet(self, image, code_ocr, number_ocr):
        try:
            temp_image_path = "temp_uploaded_image.jpg"
            image.save(temp_image_path)

            codes, arabic_names, english_names, numbers = self.ocr.process_grade_sheet(
                temp_image_path,
                code_ocr_method="ocr" if code_ocr else "training",
                number_ocr_method="ocr" if number_ocr else "training",
            )

            excel_path = self.ocr.create_excel_file(
                codes, arabic_names, english_names, numbers
            )
            return f"Excel file created: {excel_path}"
        except Exception as e:
            return f"Error processing grade sheet: {str(e)}"

    def process_bubble_sheet(self, bubble_sheet_image, model_answer_file):
        try:
            temp_bubble_sheet_path = "temp_bubble_sheet_image.jpg"
            bubble_sheet_image.save(temp_bubble_sheet_path)

            temp_model_answer_path = "temp_model_answer.txt"
            with open(temp_model_answer_path, "w") as f:
                f.write(model_answer_file.read().decode())

            img = cv2.imread(temp_bubble_sheet_path)
            model_answer = read_answers_from_file(temp_model_answer_path)

            img = cv2.resize(img, (800, 1000))
            paper = extract_paper_region(img)

            bubble_code = extract_bubble_code(paper)
            student_bubble_code_img, student_bubble_code = get_student_bubble_code(
                bubble_code
            )

            code = extract_student_code(paper)
            cropped_code = crop_code(code)
            digits = segment_id(cropped_code)
            written_code_str = "".join(get_code_prediction(digits))

            answers_region = extract_answers_region(paper)
            answers_img, answers, grades = get_student_answers(
                answers_region, model_answer
            )

            output_folder = f"outputs/{written_code_str}"
            os.makedirs(output_folder, exist_ok=True)

            cv2.imwrite(
                os.path.join(output_folder, "student_bubble_code.jpg"),
                student_bubble_code_img,
            )
            cv2.imwrite(
                os.path.join(output_folder, "student_written_code.jpg"), cropped_code
            )
            cv2.imwrite(os.path.join(output_folder, "student_answers.jpg"), answers_img)

            fill_grades_in_sheet(written_code_str, grades)

            return answers_img, f"Excel updated for student code: {written_code_str}"
        except Exception as e:
            return None, f"Error processing bubble sheet: {str(e)}"

    def create_gradio_interface(self):
        """
        Create and configure the Gradio interface.
        """
        with gr.Blocks() as interface:
            gr.Markdown("# Grade and Bubble Sheet Processor")

            with gr.Tab("Process Grade Sheet"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Grade Sheet")
                        image_input = gr.Image(type="pil", label="Grade Sheet Image")
                        code_ocr_checkbox = gr.Checkbox(label="Use Code OCR")
                        number_ocr_checkbox = gr.Checkbox(label="Use Number OCR")
                        grade_sheet_button = gr.Button("Process Grade Sheet")
                    with gr.Column(scale=1):
                        gr.Markdown("### Result")
                        grade_sheet_output = gr.Textbox(label="Result")

                grade_sheet_button.click(
                    self.process_grade_sheet,
                    inputs=[image_input, code_ocr_checkbox, number_ocr_checkbox],
                    outputs=grade_sheet_output,
                )

            with gr.Tab("Process Bubble Sheet"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Bubble Sheet")
                        bubble_sheet_image_input = gr.Image(
                            type="pil", label="Bubble Sheet Image"
                        )
                        model_answer_file_input = gr.File(
                            label="Model Answer File (.txt)"
                        )
                        bubble_sheet_button = gr.Button("Process Bubble Sheet")
                    with gr.Column(scale=1):
                        gr.Markdown("### Result")
                        bubble_sheet_output_img = gr.Image(
                            label="Corrected Bubble Sheet Image"
                        )
                        bubble_sheet_output_text = gr.Textbox(label="Result")

                bubble_sheet_button.click(
                    self.process_bubble_sheet,
                    inputs=[bubble_sheet_image_input, model_answer_file_input],
                    outputs=[bubble_sheet_output_img, bubble_sheet_output_text],
                )

        return interface


def main():
    app = GradiosApp()
    interface = app.create_gradio_interface()
    interface.launch()


if __name__ == "__main__":
    main()
