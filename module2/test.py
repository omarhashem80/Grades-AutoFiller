# import os
# import cv2
# from main import process_image_and_grades
#
# # Build the path dynamically
# script_dir = os.path.dirname(__file__)
# image_path = os.path.join(script_dir, '', 'test.jpg')
#
# # Load the image
# image = cv2.imread(image_path)
#
# process_image_and_grades(image, 'model_13.txt')



import os
import cv2
from main import process_image_and_grades

# Build the path dynamically
script_dir = os.path.dirname(__file__)
test_cases_dir = os.path.join(script_dir, 'testCases/13')

# Check if the directory exists
if os.path.isdir(test_cases_dir):
    # Loop over all files in the directory
    for file_name in os.listdir(test_cases_dir):
        # Check if the file is an image (e.g., .jpg, .png, .jpeg)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(file_name)
            image_path = os.path.join(test_cases_dir, file_name)

            # Load the image
            image = cv2.imread(image_path)

            # Process the image and grades
            process_image_and_grades(image, 'model_13.txt')
else:
    print(f"The directory {test_cases_dir} does not exist.")
