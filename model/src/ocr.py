import pytesseract
from pytesseract import Output

# Point to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(preprocessed_image):
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Using image_to_data to extract confidence scores along with text
    details = pytesseract.image_to_data(preprocessed_image, output_type=Output.DICT, config=custom_config)
    
    texts = []
    confidences = []
    
    for i, text in enumerate(details['text']):
        if text.strip():  # Ignore empty spaces
            texts.append(text.strip())
            confidences.append(int(details['conf'][i]))
            
    if texts:
        final_text = "".join(texts)
        # Calculate average confidence if multiple text blocks are detected
        avg_conf = sum(confidences) / len(confidences)
        return final_text, avg_conf
        
    return "", 0.0