import os
import cv2
import numpy as np
import joblib
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import gradio as gr
from Grade_sheet_processor import GradeSheetOCR


class GradeSheetOCRApp:
    """
    Application to process grade sheets using OCR or training-based methods.
    Allows users to upload a grade sheet image and extract student information.
    """

    def __init__(self):
        """
        Initializes the GradeSheetOCRApp with an instance of GradeSheetOCR.
        """
        self.ocr = GradeSheetOCR()

    def create_gradio_interface(self):
        """
        Creates a Gradio interface for processing grade sheets.

        Returns:
            gr.Interface: Configured Gradio interface for user interaction.
        """

        def process_grade_sheet(image, code_ocr, number_ocr):
            """
            Processes the uploaded grade sheet image and extracts information.

            Args:
                image (PIL.Image): Uploaded grade sheet image.
                code_ocr (bool): Flag to use OCR for codes.
                number_ocr (bool): Flag to use OCR for numbers.

            Returns:
                str: Result message indicating success or failure.
            """
            try:
                # Save uploaded image temporarily
                temp_image_path = "temp_uploaded_image.jpg"
                image.save(temp_image_path)

                # Process the grade sheet
                codes, arabic_names, english_names, numbers = (
                    self.ocr.process_grade_sheet(
                        temp_image_path,
                        code_ocr_method="ocr" if code_ocr else "training",
                        number_ocr_method="ocr" if number_ocr else "training",
                    )
                )

                # Create Excel file
                excel_path = self.ocr.create_excel_file(
                    codes,
                    arabic_names,
                    english_names,
                    numbers,
                )

                return f"Excel file created: {excel_path}"
            except Exception as e:
                return f"Error processing grade sheet: {str(e)}"

        # Create Gradio interface
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


def main():
    """
    Main function to launch the Gradio interface for the Grade Sheet OCR app.
    """
    app = GradeSheetOCRApp()
    interface = app.create_gradio_interface()
    interface.launch()


if __name__ == "__main__":
    main()
