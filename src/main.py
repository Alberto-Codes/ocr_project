from pdf2image import convert_from_path
import numpy as np
import cv2
import pytesseract
import re

def preprocess_image(image, debug_dir):
    os.makedirs(debug_dir, exist_ok=True)

    # Convert the PIL Image to a NumPy array and then to grayscale
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(debug_dir, '1_grayscale.png'), image_array)

    # Apply adaptive thresholding for binarization
    thresh_image = cv2.adaptiveThreshold(
        image_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
    cv2.imwrite(os.path.join(debug_dir, '2_adaptive_threshold.png'), thresh_image)

    # Apply morphological operations: Closing first to fill small holes, then opening to remove noise
    kernel_close = np.ones((3, 3), np.uint8)
    morph_close = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel_close)
    cv2.imwrite(os.path.join(debug_dir, '3_morph_close.png'), morph_close)

    kernel_open = np.ones((2, 2), np.uint8)
    morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_open)
    cv2.imwrite(os.path.join(debug_dir, '4_morph_open.png'), morph_open)

    # Apply stronger denoising to reduce background noise and enhance text
    denoised = cv2.fastNlMeansDenoising(morph_open, h=30, templateWindowSize=7, searchWindowSize=21)
    cv2.imwrite(os.path.join(debug_dir, '5_denoised.png'), denoised)

    # Background subtraction using GaussianBlur instead of medianBlur for smoother results
    bg = cv2.GaussianBlur(denoised, (21, 21), 0)
    cv2.imwrite(os.path.join(debug_dir, '6_background.png'), bg)

    diff_image = cv2.absdiff(denoised, bg)
    diff_image = 255 - diff_image
    cv2.imwrite(os.path.join(debug_dir, '7_background_subtracted.png'), diff_image)

    # Normalize the image to enhance contrast
    norm_image = cv2.normalize(diff_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(debug_dir, '8_normalized.png'), norm_image)

    # Apply Otsu's thresholding for final binarization
    _, result = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_dir, '9_final_otsu.png'), result)

    return result

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
