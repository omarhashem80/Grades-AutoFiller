import cv2
import numpy as np
import joblib
import pytesseract
import heapq
from skimage.feature import hog


class ImageProcessor:

    def __init__(self, tesseract_path=r"/usr/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.load_digit_model = joblib.load("Module1/svm_model_digits.joblib")
        self.load_symbol_model = joblib.load("Module1/svm_model_symbols.joblib")

    def read_image(self, image_path):
        return cv2.imread(image_path)

    def convert_to_grayscale(self, image):
        if len(image.shape) == 2:
            return image
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

    def extract_hog_144_features(self, image, target_size=(28, 28)):
        image = cv2.resize(image, dsize=target_size)
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L2-Hys",
            visualize=False,
        )
        return features

    def extract_hog_756_features(self, image, target_size=(32, 64)):
        image = cv2.resize(image, dsize=target_size)
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L2-Hys",
            visualize=False,
        )
        return features

    def extract_hog_1764_features(self, image, target_size=(32, 32)):
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
        hog_features = self.extract_hog_1764_features(image)
        hog_features = hog_features.reshape(1, -1)
        predicted_digit = self.load_digit_model.predict(hog_features)
        return predicted_digit[0]

    def predict_symbol(self, image):
        hog_features = self.extract_hog_1764_features(image)
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
