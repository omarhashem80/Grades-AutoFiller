import cv2
import numpy as np


def reorder_corners(corners):
    """
    Reorders the corners of a quadrilateral in a consistent order.

    Args:
        corners (numpy.ndarray): A 4x2 array of corner points.

    Returns:
        numpy.ndarray: An array with the corners rearranged in the following order:
                       top-left, top-right, bottom-right, bottom-left.
    """
    ordered_corners = np.zeros((4, 2), dtype="float32")
    corner_sums = corners.sum(axis=1)
    corner_diffs = np.diff(corners, axis=1)

    ordered_corners[0] = corners[np.argmin(corner_sums)]   # Top-left
    ordered_corners[1] = corners[np.argmin(corner_diffs)] # Top-right
    ordered_corners[2] = corners[np.argmax(corner_sums)]  # Bottom-right
    ordered_corners[3] = corners[np.argmax(corner_diffs)] # Bottom-left

    return ordered_corners


def apply_perspective_transform(image, points):
    """
    Applies a perspective transform to an image using four corner points.

    Args:
        image (numpy.ndarray): The input image.
        points (numpy.ndarray): A 4x2 array of corner points.

    Returns:
        numpy.ndarray: The perspective-transformed image.
    """
    ordered_points = reorder_corners(points)
    (top_left, top_right, bottom_right, bottom_left) = ordered_points

    width_top = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    height_right = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_left = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_width = max(int(width_top), int(width_bottom))
    max_height = max(int(height_right), int(height_left))

    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
    transformed_image = cv2.warpPerspective(image, perspective_matrix, (max_width, max_height))

    return transformed_image


def extract_paper_region(image):
    """
    Extracts the largest quadrilateral region resembling a paper sheet from an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The extracted paper region as an image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

            if len(approximated_contour) == 4 and cv2.contourArea(contour) > 0.2 * image_area:
                return apply_perspective_transform(image, approximated_contour.reshape(4, 2))

    return None
