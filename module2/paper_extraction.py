import numpy as np
import cv2


def arrange_corners(corners):
    """
    Orders the four corners of a quadrilateral in the following order:
    top-left, top-right, bottom-right, bottom-left.

    Args:
        corners (numpy.ndarray): A 4x2 array containing the (x, y) coordinates of the quadrilateral's corners.

    Returns:
        numpy.ndarray: A 4x2 array with the corners arranged in the specified order.
    """
    # Initialize a zero array to store the ordered corners
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of x and y coordinates to find the top-left and bottom-right corners
    total = corners.sum(axis=1)
    # Difference between x and y coordinates to find the top-right and bottom-left corners
    diff = np.diff(corners, axis=1)

    rect[0] = corners[np.argmin(total)]  # Top-left corner
    rect[1] = corners[np.argmin(diff)]   # Top-right corner
    rect[2] = corners[np.argmax(total)]  # Bottom-right corner
    rect[3] = corners[np.argmax(diff)]   # Bottom-left corner

    return rect


def image_transform(image, pts):
    """
    Transforms the perspective of an image to a top-down view of the region defined by the given points.

    Args:
        image (numpy.ndarray): The input image.
        pts (numpy.ndarray): A 4x2 array of (x, y) coordinates representing the corners of the quadrilateral region.

    Returns:
        numpy.ndarray: The warped image with the perspective transformed.
    """
    # Arrange the corners of the quadrilateral
    rect = arrange_corners(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the quadrilateral
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # Maximum width and height for the new image
    max_width = max(int(width1), int(width2))
    max_height = max(int(height1), int(height2))

    # Define the destination rectangle for the perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype=np.float32)

    # Compute the perspective transform matrix and apply it
    pers_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped_img = cv2.warpPerspective(image, pers_matrix, (max_width, max_height))

    return warped_img


def extract_paper(image):
    """
    Detects and extracts a quadrilateral-shaped object (e.g., a piece of paper) from an image.
    If found, applies a perspective transformation to provide a top-down view of the object.

    Args:
        image (numpy.ndarray): The input image in BGR format.

    Returns:
        numpy.ndarray: The perspective-transformed image of the detected object (paper).
        Returns None if no suitable quadrilateral object is detected.
    """
    # Step 1: Convert the input image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply adaptive thresholding to enhance edges
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Step 3: Find all contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Ensure there are contours to process
    if len(contours) > 0:
        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Calculate the total area of the image
        target_area = image.shape[0] * image.shape[1]

        # Iterate through the sorted contours
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)  # Allowable error margin
            paper_contour = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has 4 corners and significant area
            if len(paper_contour) == 4 and cv2.contourArea(contour) > 0.2 * target_area:
                # Perform a perspective transformation to extract the paper
                paper = image_transform(image, paper_contour.reshape(4, 2))
                return paper  # Return the transformed image

    # Return None if no suitable quadrilateral is detected
    return None
