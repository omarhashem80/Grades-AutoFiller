import os
import cv2
from main import process_image_and_grades

# Build the path dynamically
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, '', 'test.jpg')

# Load the image
image = cv2.imread(image_path)

process_image_and_grades(image, 'model_test.txt')



