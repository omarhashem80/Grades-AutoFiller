# 📊 Grades Auto-Filler: A Smart Grading Assistant

## Overview
Grades Auto-Filler is a cutting-edge project designed to revolutionize grading for TAs and professors. This tool simplifies and automates the grading process by handling electronic grade sheets and correcting MCQ bubble sheet exams. With its user-friendly interface and robust processing capabilities, Grades Auto-Filler ensures efficiency, accuracy, and adaptability to varying input formats.

---

## 🌟 Key Features

### **Module 1: Smart Grade Sheet Processor**
- **Handwritten Grade Recognition:** Transforms handwritten grades from photos into electronic records.
- **Robust to Variations:**
    - Handles skewed, scaled, and rotated images (except upside-down).
    - Adapts to different ink colors, handwriting styles, and grade sheet formats.
    - Accurately processes numeric values, special symbols (✓, ✗, -, etc.), and tally marks.
- **Intelligent OCR:** Allows users to choose between pre-built OCR tools or custom classifiers.
- **Output:** Automatically generates a clean and organized Excel sheet.

### **Module 2: MCQ Bubble Sheet Corrector**
- **Adaptive to Formats:** Processes bubble sheets with varying numbers of questions and choices.
- **Model Answer Integration:** Cross-references with a model answer to mark responses.
- **Flexible Grading:** Customizable grading logic (e.g., question weightage, penalties).
- **Output:** Generates a detailed results spreadsheet with correct and incorrect answers clearly marked.

---

## 🚀 How It Works

1. **Grade Sheet Processing:**
    - Capture a photo of the grades sheet.
    - Upload the image to the system.
    - Choose OCR mode (pre-built or custom classifier).
    - Receive a fully populated Excel file.

2. **Bubble Sheet Correction:**
    - Scan or photograph the bubble sheet.
    - Provide the model answers in a simple text file.
    - Specify grading parameters (e.g., question value, penalty).
    - Receive a detailed results spreadsheet.

---

## 💡 Why Choose Grades Auto-Filler?

- **Time-Saving:** Automates tedious manual grading tasks.
- **Accurate:** Ensures minimal errors in grade transcription and bubble sheet correction.
- **Flexible:** Adapts to diverse formats and grading schemes.
- **Scalable:** Supports varying class sizes and grading needs.

---

## 🔧 Setup and Usage
1. **Dependencies:** Ensure you have the following installed:
   - Python 
   - Required libraries (`OCR`, `Image Processing`, etc.)
   - Install the required libraries with the following command:

      ```bash
      pip install -r requirements.txt
      ```

   - Spreadsheet tools (e.g., Excel support library)
2. **Run the Program:**
    - Clone this repository.
    - Run the script for your chosen module.
    - Follow the prompts to input files and parameters.
3. **View Results:** The output will be saved as an Excel file.

---

## 🛠️ Future Enhancements
- AI-powered handwriting recognition for improved accuracy.
- Support for online integration with Learning Management Systems (LMS).
- Enhanced processing speed for large datasets.

---

## 🏆 Contributors
This project was developed by a dedicated team committed to simplifying grading processes in educational institutions.
- [Omar Hashem](omarhashem80)
---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

For any queries or contributions, feel free to open an issue or submit a pull request.

---


