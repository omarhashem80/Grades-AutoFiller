def negative_transformation(image):
    """
    Applies negative transformation to the input image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Negative-transformed image.
    """
    return 255 - image


def extract_answers_region(paper_image):
    """
    Extracts the region of interest containing the answers from the paper.

    Args:
        paper_image (numpy.ndarray): Input scanned paper image.

    Returns:
        numpy.ndarray: Cropped region containing the answers.
    """
    height, _ = paper_image.shape[:2]
    cropped_height = height // 3
    return paper_image[cropped_height:, :]


def extract_bubble_code(paper):
    """
    Extracts the bubble code area from the given paper image.

    Parameters:
        paper (numpy.ndarray): The scanned image of the paper.

    Returns:
        numpy.ndarray: The extracted segment containing the bubble code.
    """
    x, y = paper.shape[:2]
    # Determine the region of interest
    new_x = (x // 3) + 10
    new_y = (y // 2) + 40
    segment = paper[:new_x, :new_y]

    return segment
