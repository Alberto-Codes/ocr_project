import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np

# Path to the PDF file
pdf_path = 'path_to_your_pdf.pdf'

# Convert PDF to images
pages = convert_from_path(pdf_path, dpi=300)

# Iterate over each page
for page_number, page in enumerate(pages, start=1):
    # Convert PIL image to OpenCV format
    image = np.array(page)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Apply Gaussian Blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(blurred)

    # Step 3: Otsu's thresholding after contrast enhancement
    _, binary_image = cv2.threshold(enhanced_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological closing to strengthen characters
    kernel = np.ones((2, 2), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Step 5: OCR with Tesseract
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(closed_image, config=custom_config)

    # Save the extracted text
    with open(f'page_{page_number}.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(extracted_text)
