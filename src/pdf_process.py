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

    # Apply Otsu's thresholding to binarize the image
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Reduce noise with bilateral filtering
    denoised_image = cv2.bilateralFilter(binary_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Morphological opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)  # Kernel size can be adjusted
    clean_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)

    # OCR with Tesseract
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(clean_image, config=custom_config)

    # Save the extracted text
    with open(f'page_{page_number}.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(extracted_text)
