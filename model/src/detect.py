import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="weights/plate_detector.pt"):
        self.model = YOLO(model_path)
        
    def detect_and_crop(self, image_path):
        original_image = cv2.imread(image_path)
        if original_image is None:
            return None, []
            
        results = self.model(original_image, verbose=False)
        cropped_plates = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_plate = original_image[y1:y2, x1:x2]
                
                if cropped_plate.size != 0:
                    cropped_plates.append((cropped_plate, (x1, y1, x2, y2)))
                    
        return original_image, cropped_plates