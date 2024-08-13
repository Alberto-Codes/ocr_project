import cv2
import numpy as np
from PIL import Image
import pytesseract
from spellchecker import SpellChecker
import re

def preprocess_image(image):
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_image = clahe.apply(gray_image)
    
    # Try different thresholding methods
    _, binary_image = cv2.threshold(contrast_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise (adjust parameters as needed)
    denoised_image = cv2.fastNlMeansDenoising(binary_image, h=10)
    
    # Deskew
    coords = np.column_stack(np.where(denoised_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = denoised_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_image = cv2.warpAffine(denoised_image, M, (w, h),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed_image

def extract_text_from_image(image):
    # Custom config with whitelist
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:$.,/-'
    
    # Extract text
    text = pytesseract.image_to_string(image, config=custom_config)
    
    return text

def extract_micr(image):
    # Assuming MICR is at the bottom 15% of the image
    h, w = image.shape
    bottom = int(h * 0.85)
    micr_region = image[bottom:h, :]
    
    # OCR with MICR-specific config
    micr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:$.,/-'
    micr_text = pytesseract.image_to_string(micr_region, config=micr_config)
    
    return micr_text

def correct_spelling(text):
    spell = SpellChecker()
    corrected_text = []
    for word in text.split():
        corrected_text.append(spell.correction(word))
    
    return ' '.join(corrected_text)

def extract_check_info(text):
    # Define regex patterns for check-specific information
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b'
    amount_pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'
    
    # Extract information
    date = re.search(date_pattern, text)
    amount = re.search(amount_pattern, text)
    
    return {
        'date': date.group() if date else None,
        'amount': amount.group() if amount else None
    }

# Main function
def process_check(image_path):
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    
    extracted_text = extract_text_from_image(processed_image)
    micr_text = extract_micr(processed_image)
    
    corrected_text = correct_spelling(extracted_text)
    
    check_info = extract_check_info(corrected_text)
    
    return {
        'full_text': corrected_text,
        'micr': micr_text,
        'date': check_info['date'],
        'amount': check_info['amount']
    }

# Usage
image_path = './data/check_image.jpg'
result = process_check(image_path)
print(result)
