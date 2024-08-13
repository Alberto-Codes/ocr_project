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

    # Apply a gentle Gaussian blur to reduce noise but keep edges
    blurred_image = cv2.GaussianBlur(image_array, (5, 5), 0)
    cv2.imwrite(os.path.join(debug_dir, '2_blurred.png'), blurred_image)

    # Use edge detection to find strong edges, which typically include text
    edges = cv2.Canny(blurred_image, 50, 150)
    cv2.imwrite(os.path.join(debug_dir, '3_edges.png'), edges)

    # Dilate the edges to make them more pronounced
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(os.path.join(debug_dir, '4_dilated_edges.png'), dilated_edges)

    # Find contours to isolate areas of interest (i.e., text regions)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to isolate text regions
    mask = np.zeros_like(image_array)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    cv2.imwrite(os.path.join(debug_dir, '5_contours_mask.png'), mask)

    # Apply the mask to the original image to isolate the text areas
    masked_image = cv2.bitwise_and(image_array, image_array, mask=mask)
    cv2.imwrite(os.path.join(debug_dir, '6_masked_image.png'), masked_image)

    # Apply bilateral filtering to reduce noise but keep edges sharp
    bilateral_filtered = cv2.bilateralFilter(masked_image, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(os.path.join(debug_dir, '7_bilateral_filtered.png'), bilateral_filtered)

    # Enhance contrast of the bilateral filtered image
    contrast_enhanced = cv2.normalize(bilateral_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(debug_dir, '8_contrast_enhanced.png'), contrast_enhanced)

    # Use contour filtering to remove small noise dots that are not part of text
    contours, _ = cv2.findContours(contrast_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # Filter out small contours that are likely noise
            cv2.drawContours(contrast_enhanced, [cnt], -1, (0), thickness=cv2.FILLED)
    cv2.imwrite(os.path.join(debug_dir, '9_filtered_noise.png'), contrast_enhanced)

    # Apply Otsu's thresholding for final binarization
    _, final_threshold = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_dir, '10_final_threshold.png'), final_threshold)

    return final_threshold

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
