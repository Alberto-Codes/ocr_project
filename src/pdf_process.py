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

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert image if text is white on black
    if np.mean(binary_image) > 127:
        binary_image = cv2.bitwise_not(binary_image)

    # Apply OCR
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(binary_image, config=custom_config)

    # Save or process the extracted text
    with open(f'page_{page_number}.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(extracted_text)
