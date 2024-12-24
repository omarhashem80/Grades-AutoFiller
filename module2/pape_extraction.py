import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def arrange_corners(corners):
    """
    Rearrange the corners of a quadrilateral to a consistent order:
    top-left, top-right, bottom-right, and bottom-left.

    Args:
        corners (ndarray): Array of shape (4, 2) representing the corner points.

    Returns:
        ndarray: Rearranged corner points in a consistent order.
    """
    # Initialize the rectangle to store ordered corners
    ordered_corners = np.zeros((4, 2), dtype="float32")

    # Compute the sum and difference of the corner coordinates
    total_sum = corners.sum(axis=1)  # Sum of x and y coordinates
    difference = np.diff(corners, axis=1)  # Difference between x and y coordinates

    # Assign corners based on specific properties
    ordered_corners[0] = corners[np.argmin(total_sum)]  # Top-left has the smallest sum
    ordered_corners[1] = corners[np.argmin(difference)]  # Top-right has the smallest difference
    ordered_corners[2] = corners[np.argmax(total_sum)]  # Bottom-right has the largest sum
    ordered_corners[3] = corners[np.argmax(difference)]  # Bottom-left has the largest difference

    return ordered_corners


def image_transform(image, points):
    """
    Perform a perspective transformation on the input image to a top-down view.

    Args:
        image (ndarray): The input image.
        points (ndarray): Array of corner points to define the transformation.

    Returns:
        ndarray: The warped image with a top-down perspective.
    """
    # Arrange the corner points in a consistent order
    rect = arrange_corners(points)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # Compute the width of the transformed image
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = max(int(width_top), int(width_bottom))

    # Compute the height of the transformed image
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    max_height = max(int(height_left), int(height_right))

    # Define the destination points for the transformed image
    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(rect, destination_points)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    return warped_image


def extract_paper(image):
    """
    Extract a rectangular region resembling a paper from the image using contour detection.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The extracted paper region as a warped image.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to detect edges
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # plt.imshow(binary_image, cmap='gray')
    # plt.show()
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Proceed if at least one contour is found
    if contours:
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Calculate the minimum acceptable area for a valid paper region
        min_area = 0.2 * (image.shape[0] * image.shape[1])

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour has 4 points and is sufficiently large
            if len(approximated_contour) == 4 and cv2.contourArea(contour) > min_area:
                # Transform the detected paper region to a top-down view
                paper_image = image_transform(image, approximated_contour.reshape(4, 2))
                return paper_image

    return None
