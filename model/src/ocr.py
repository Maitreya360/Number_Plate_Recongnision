import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(preprocessed_image):
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    details = pytesseract.image_to_data(
        preprocessed_image, 
        output_type=Output.DICT, 
        config=custom_config, 
        lang='eng' 
    )
    
    texts = []
    confidences = []
    
    for i, text in enumerate(details['text']):
        conf = int(details['conf'][i])
        if text.strip() and conf > 0:
            texts.append(text.strip())
            confidences.append(conf)
            
    if texts:
        final_text = "".join(texts)
        avg_conf = sum(confidences) / len(confidences)
        return final_text, avg_conf
        
    return "", 0.0