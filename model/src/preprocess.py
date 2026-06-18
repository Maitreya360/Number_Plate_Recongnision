import cv2

def clean_plate_image(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    filtered = cv2.bilateralFilter(upscaled, 3, 17, 17)
    
    _, binarized = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    padded_plate = cv2.copyMakeBorder(
        binarized, 20, 20, 20, 20, 
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    return padded_plate