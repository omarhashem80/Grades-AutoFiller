import os
import cv2
import numpy as np
import joblib
import pandas as pd
import pytesseract
import gradio as gr
import heapq


class ImageProcessor:
    def __init__(self, tesseract_path=r"/usr/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.load_digit_model = joblib.load(
            "/home/a7med/Desktop/Grade-auto-filler/src/app/Backend/Module1/svm_model_digits.joblib"
        )
        self.load_symbol_model = joblib.load(
            "/home/a7med/Desktop/Grade-auto-filler/src/app/Backend/Module1/svm_model_symbols.joblib"
        )

    def read_image(self, image_path):
        return cv2.imread(image_path)

    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self, grayscale_image):
        _, thresholded_image = cv2.threshold(
            grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded_image

    def invert_image(self, thresholded_image):
        return cv2.bitwise_not(thresholded_image)

    def dilate_image(self, image, kernel=None, iterations=5):
        return cv2.dilate(image, kernel, iterations=iterations)

    def find_contours(self, dilated_image, method=cv2.RETR_TREE):
        contours, _ = cv2.findContours(dilated_image, method, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def extract_hog_features(self, image, target_size=(32, 32)):
        image = cv2.resize(image, dsize=target_size)
        window_size = target_size
        cell_size = (4, 4)
        block_size_in_cells = (2, 2)
        block_size = (
            block_size_in_cells[1] * cell_size[1],
            block_size_in_cells[0] * cell_size[0],
        )
        block_stride = (cell_size[1], cell_size[0])
        nbins = 9
        hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, nbins)
        h = hog.compute(image)
        return h.flatten()

    def predict_digit(self, image):
        hog_features = self.extract_hog_features(image)
        hog_features = hog_features.reshape(1, -1)
        predicted_digit = self.load_digit_model.predict(hog_features)
        return predicted_digit[0]

    def predict_symbol(self, image):
        hog_features = self.extract_hog_features(image)
        hog_features = hog_features.reshape(1, -1)
        predicted_symbol = self.load_symbol_model.predict(hog_features)
        return predicted_symbol[0]

    def filter_rectangular_contours(self, contours):
        rectangular_contours = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if h > 500 and w > 500:
                    rectangular_contours.append(approx)
        return rectangular_contours

    def find_third_largest_contour(self, rectangular_contours):
        contours_with_areas = [
            (contour, cv2.contourArea(contour)) for contour in rectangular_contours
        ]
        if len(contours_with_areas) < 3:
            return contours_with_areas[0][0], contours_with_areas[0][1]
        top_3_contours = heapq.nlargest(3, contours_with_areas, key=lambda x: x[1])
        if (5e5 - top_3_contours[2][1]) > 3e5:
            return top_3_contours[0][0], top_3_contours[0][1]
        return top_3_contours[2][0], top_3_contours[2][1]

    def order_points(self, pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process_image(self, image_path):
        image = self.read_image(image_path)
        image = cv2.resize(image, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)

        grayscale_image = self.convert_to_grayscale(image)
        thresholded_image = self.threshold_image(grayscale_image)
        inverted_image = self.invert_image(thresholded_image)
        dilated_image = self.dilate_image(inverted_image)

        contours = self.find_contours(dilated_image)
        rectangular_contours = self.filter_rectangular_contours(contours)
        third_largest_contour, _ = self.find_third_largest_contour(rectangular_contours)

        ordered_points = self.order_points(third_largest_contour)

        # Calculate distances for aspect ratio
        distance_top_left_to_top_right = np.linalg.norm(
            ordered_points[0] - ordered_points[1]
        )
        distance_top_left_to_bottom_left = np.linalg.norm(
            ordered_points[0] - ordered_points[3]
        )

        aspect_ratio = distance_top_left_to_bottom_left / distance_top_left_to_top_right
        existing_width = image.shape[1]
        new_width = int(existing_width * 0.9)
        new_height = int(new_width * aspect_ratio)

        # Perspective transform
        pts1 = np.float32(ordered_points)
        pts2 = np.float32(
            [[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_corrected_image = cv2.warpPerspective(
            image, matrix, (new_width, new_height)
        )

        return perspective_corrected_image


class GradeSheetOCR:
    def __init__(self):
        self.image_processor = ImageProcessor()

    def _process_code_ocr(self, code_image):
        min_crop_size = (35, 100)
        processed_image = self._remove_table_lines(code_image)
        kernel = np.ones((2, 5), np.uint8)
        dilated_image = cv2.dilate(processed_image, kernel, iterations=10)
        contours = self.image_processor.find_contours(dilated_image)
        bounding_boxes = self._convert_contours_to_bounding_boxes(contours)
        min_height = self._get_min_height_of_bounding_boxes(bounding_boxes)
        rows = self._club_bounding_boxes_into_rows(bounding_boxes, min_height)
        self._sort_rows_by_x_coordinate(rows)
        codes = []

        for row in rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                cropped_image = code_image[y : y + h, x : x + w]
                w -= 10
                if (
                    cropped_image.shape[0] < min_crop_size[0]
                    or cropped_image.shape[1] < min_crop_size[1] - 10
                ):
                    continue
                cropped_image = cv2.resize(
                    cropped_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
                )

                if cropped_image.shape[0] < 60 or cropped_image.shape[1] < 150:
                    continue
                codes.append(pytesseract.image_to_string(cropped_image).strip())
        return codes

    def _process_code_training(self, code_image):
        code_image = self.image_processor.invert_image(
            self.image_processor.threshold_image(
                self.image_processor.convert_to_grayscale(code_image)
            )
        )
        code_image = cv2.erode(code_image, np.ones((2, 1), np.uint8), iterations=7)
        contours = self.image_processor.find_contours(code_image)
        bounding_boxes = self._convert_contours_to_bounding_boxes(contours)
        min_height = self._get_min_height_of_bounding_boxes(bounding_boxes)
        rows = self._club_bounding_boxes_into_rows(bounding_boxes, min_height)
        self._sort_rows_by_x_coordinate(rows)
        predicted_number = []

        for row in rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                cropped_image = code_image[y : y + h, x : x + w]
                if cropped_image.shape[0] < 20 or cropped_image.shape[1] < 20:
                    continue
                predicted_number.append(
                    self.image_processor.predict_digit(cropped_image)
                )
        return "".join(map(str, predicted_number))

    def _process_names_ocr(self, image):
        min_crop_size = (35, 100)
        processed_image = self._remove_table_lines(image)
        processed_image = cv2.GaussianBlur(processed_image, (5, 5), 1)
        kernel = np.ones((2, 5), np.uint8)
        dilated_image = cv2.dilate(processed_image, kernel, iterations=10)
        contours = self.image_processor.find_contours(dilated_image)
        bounding_boxes = self._convert_contours_to_bounding_boxes(contours)
        min_height = self._get_min_height_of_bounding_boxes(bounding_boxes)
        rows = self._club_bounding_boxes_into_rows(bounding_boxes, min_height)
        self._sort_rows_by_x_coordinate(rows)
        names = []

        for row in rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                cropped_image = image[y : y + h, x : x + w]
                w -= 10
                if (
                    cropped_image.shape[0] < min_crop_size[0]
                    or cropped_image.shape[1] < min_crop_size[1] - 10
                ):
                    continue
                cropped_image = cv2.resize(
                    cropped_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
                )

                if cropped_image.shape[0] < 60 or cropped_image.shape[1] < 150:
                    continue
                names.append(pytesseract.image_to_string(cropped_image).strip())
        return names

    def _process_digit_cell_ocr(self, image):
        min_crop_size = (35, 100)
        kernel = np.ones((2, 5), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=10)
        contours = self.image_processor.find_contours(dilated_image)
        bounding_boxes = self._convert_contours_to_bounding_boxes(contours)
        min_height = self._get_min_height_of_bounding_boxes(bounding_boxes)
        rows = self._club_bounding_boxes_into_rows(bounding_boxes, min_height)
        self._sort_rows_by_x_coordinate(rows)
        numbers = []

        for row in rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                cropped_image = image[y : y + h, x : x + w]
                w -= 10
                if (
                    cropped_image.shape[0] < min_crop_size[0]
                    or cropped_image.shape[1] < min_crop_size[1] - 10
                ):
                    continue
                cropped_image = cv2.resize(
                    cropped_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
                )

                if cropped_image.shape[0] < 60 or cropped_image.shape[1] < 150:
                    continue
                cropped_image = self.image_processor.invert_image(
                    self.image_processor.threshold_image(
                        self.image_processor.convert_to_grayscale(cropped_image)
                    )
                )
                numbers.append(
                    pytesseract.image_to_string(
                        cropped_image,
                        config="-l osd --psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",
                    ).strip()
                )
        return numbers

    def _process_digit_cell_training(self, image):
        processed_image = self._remove_table_lines(image)
        kernel = np.ones((2, 5), np.uint8)
        dilated_image = cv2.dilate(processed_image, kernel, iterations=10)
        contours = self.image_processor.find_contours(dilated_image)
        bounding_boxes = self._convert_contours_to_bounding_boxes(contours)
        min_height = self._get_min_height_of_bounding_boxes(bounding_boxes)
        rows = self._club_bounding_boxes_into_rows(bounding_boxes, min_height)
        recognized_digits = []

        for row in rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                cropped_image = image[y : y + h, x : x + w]
                if cropped_image.shape[0] < 50 or cropped_image.shape[1] < 50:
                    continue
                digit = self.image_processor.predict_digit(cropped_image)
                recognized_digits.append(digit)

        return recognized_digits

    def _process_image_into_shapes_cells(self, image):
        height, width = image.shape[:2]
        rows, columns = 17, 2
        cell_height = height // rows
        cell_width = width // columns
        all_results = []

        for row in range(rows):
            row_results = []
            for col in range(columns):
                start_y = row * cell_height + 10
                end_y = (row + 1) * cell_height - 10
                start_x = col * cell_width + 10
                end_x = (col + 1) * cell_width - 10

                cell = image[start_y:end_y, start_x:end_x]
                result = self._process_shape(self.image_processor.predict_symbol(cell))
                row_results.append(result)

            all_results.append(row_results)
        return all_results

    def _process_shape(self, s):
        if s == "box":
            return 0
        elif s == "correct":
            return 5
        elif s == "empty":
            return -1
        elif s.startswith("horizontal") and s[-1].isdigit():
            if int(s[-1]) == 1:
                return 0
            return 5 - int(s[-1])
        elif s.startswith("vertical") and s[-1].isdigit():
            return int(s[-1])
        elif s == "question":
            return -2
        else:
            return None

    def _split_image_sections(self, corrected_image):
        return {
            "code": corrected_image[:, : int(corrected_image.shape[1] * 0.1)],
            "arabic_name": corrected_image[
                int(corrected_image.shape[1] * 0.1) : int(
                    corrected_image.shape[1] * 0.34
                )
            ],
            "english_name": corrected_image[
                int(corrected_image.shape[1] * 0.34) : int(
                    corrected_image.shape[1] * 0.7
                )
            ],
            "grade_numerical": corrected_image[
                int(corrected_image.shape[1] * 0.7) : int(
                    corrected_image.shape[1] * 0.81
                )
            ],
            "grade_shapes": corrected_image[int(corrected_image.shape[1] * 0.81) :],
        }

    def _remove_table_lines(self, image):
        grey = self.image_processor.convert_to_grayscale(image)
        thresholded_image = self.image_processor.threshold_image(grey)
        inverted_image = self.image_processor.invert_image(thresholded_image)

        # Vertical line removal
        ver = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
        vertical_lines_eroded = cv2.erode(inverted_image, ver, iterations=10)
        vertical_lines_eroded = cv2.dilate(vertical_lines_eroded, ver, iterations=10)

        # Horizontal line removal
        hor = np.array([[1, 1, 1, 1, 1, 1]])
        horizontal_lines_eroded = cv2.erode(inverted_image, hor, iterations=10)
        horizontal_lines_eroded = cv2.dilate(
            horizontal_lines_eroded, hor, iterations=10
        )

        # Combine and remove
        combined_image = cv2.add(vertical_lines_eroded, horizontal_lines_eroded)
        combined_image_dilated = self.image_processor.dilate_image(combined_image)
        image_without_lines = cv2.subtract(inverted_image, combined_image_dilated)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded_image = cv2.erode(image_without_lines, kernel, iterations=1)
        return self.image_processor.dilate_image(eroded_image, kernel, iterations=1)

    def _convert_contours_to_bounding_boxes(self, contours):
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            y -= 5
            h += 10
            bounding_boxes.append((x, y, w, h))
        return bounding_boxes

    def _get_min_height_of_bounding_boxes(self, bounding_boxes):
        heights = [h for _, _, _, h in bounding_boxes]
        return np.min(heights)

    def _club_bounding_boxes_into_rows(self, bounding_boxes, mean_height):
        half_mean_height = mean_height / 2
        rows = []
        current_row = [bounding_boxes[0]]
        for bounding_box in bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(
                current_bounding_box_y - previous_bounding_box_y
            )
            if distance_between_bounding_boxes <= half_mean_height:
                current_row.append(bounding_box)
            else:
                rows.append(current_row)
                current_row = [bounding_box]
        rows.append(current_row)
        return rows

    def _sort_rows_by_x_coordinate(self, rows):
        for row in rows:
            row.sort(key=lambda x: x[0])

    def _combine_number_with_shapes(self, numbers, shapes):
        combined_results = []
        # Ensure numbers and shapes have the same length
        max_length = min(len(numbers), len(shapes))

        for i in range(max_length):
            row_numbers = numbers[i] if isinstance(numbers[i], list) else [numbers[i]]
            combined_results.append(row_numbers + shapes[i])

        return combined_results

    def process_grade_sheet(
        self, image_path, code_ocr_method="training", number_ocr_method="training"
    ):
        # Process the image
        corrected_image = self.image_processor.process_image(image_path)

        # Split image into sections
        sections = self._split_image_sections(corrected_image)

        # Process each section
        codes = (
            self._process_code_ocr(sections["code"])
            if code_ocr_method == "ocr"
            else self._process_code_training(sections["code"])
        )

        arabic_names = self._process_names_ocr(sections["arabic_name"])
        english_names = self._process_names_ocr(sections["english_name"])

        numbers = (
            self._process_digit_cell_ocr(sections["grade_numerical"])
            if number_ocr_method == "ocr"
            else self._process_digit_cell_training(sections["grade_numerical"])
        )

        shapes = self._process_image_into_shapes_cells(sections["grade_shapes"])

        # Combine results
        combined_numbers = self._combine_number_with_shapes(numbers, shapes)

        return codes, arabic_names, english_names, combined_numbers

    def create_excel_file(
        self, codes, arabic_names, english_names, numbers, output_filename="output.xlsx"
    ):
        # Ensure all lists have the same length
        min_length = min(
            len(codes), len(arabic_names), len(english_names), len(numbers)
        )

        rows = []
        for i in range(min_length):
            row = {
                "Code": codes[i],
                "Student Name": arabic_names[i],
                "English Name": english_names[i],
            }

            for j, value in enumerate(numbers[i], start=1):
                if value == -1:
                    row[str(j)] = ""
                elif value == -2:
                    row[str(j)] = -2
                else:
                    row[str(j)] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(output_filename, index=False)
        print(f"Excel file '{output_filename}' created successfully!")
        return output_filename


class GradeSheetOCRApp:
    def __init__(self):
        self.ocr = GradeSheetOCR()

    def create_gradio_interface(self):
        def process_grade_sheet(image, code_ocr, number_ocr):
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
                    codes, arabic_names, english_names, numbers
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
    app = GradeSheetOCRApp()
    interface = app.create_gradio_interface()
    interface.launch()


if __name__ == "__main__":
    main()
