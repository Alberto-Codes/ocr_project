from pdf2image import convert_from_path
import numpy as np
import cv2
import pytesseract
import re

def preprocess_image(image):
    # Convert the PIL Image to a NumPy array (required by OpenCV)
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(
        image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh_image, h=10)
    
    return denoised

def extract_text_from_image(image):
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:$.,/-'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def extract_check_data(text):
    amount_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    check_number_pattern = r'\bCheck\s+No\.?\s*(\d+)\b'
    account_pattern = r'\bAccount\s+No\.?\s*(\d+)\b'
    payee_pattern = r'Pay\s+to\s+the\s+order\s+of\s+([A-Za-z\s]+)'
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b'

    amount_match = re.search(amount_pattern, text)
    check_number_match = re.search(check_number_pattern, text)
    account_match = re.search(account_pattern, text)
    payee_match = re.search(payee_pattern, text)
    date_match = re.search(date_pattern, text)

    data = {
        'amount': amount_match.group(1) if amount_match else None,
        'check_number': check_number_match.group(1) if check_number_match else None,
        'account_number': account_match.group(1) if account_match else None,
        'payee': payee_match.group(1).strip() if payee_match else None,
        'date': date_match.group() if date_match else None
    }
    return data

# Extract images from the PDF
pdf_path = './data/document_filename.pdf'
images = convert_from_path(pdf_path)

# Process each image from the PDF
for i, image in enumerate(images):
    processed_image = preprocess_image(image)
    extracted_text = extract_text_from_image(processed_image)
    check_data = extract_check_data(extracted_text)
    print(f'Extracted Data from page {i+1}: {check_data}')

# Optionally, you can save the processed image to check the preprocessing result
# cv2.imwrite(f'processed_image_{i+1}.png', processed_image)
