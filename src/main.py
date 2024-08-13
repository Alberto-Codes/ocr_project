def preprocess_image(image):
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    thresh_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised_image = cv2.fastNlMeansDenoising(thresh_image, h=30)
    
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
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def correct_spelling(text):
    from spellchecker import SpellChecker
    spell = SpellChecker()

    corrected_text = []
    for word in text.split():
        corrected_text.append(spell.correction(word))
    
    return ' '.join(corrected_text)

# Main function example
image_path = './data/check_image.jpg'
image = Image.open(image_path)
processed_image = preprocess_image(image)
extracted_text = extract_text_from_image(processed_image)
corrected_text = correct_spelling(extracted_text)

print(corrected_text)
