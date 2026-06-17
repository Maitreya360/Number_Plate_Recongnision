import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="weights/yolov8n.pt"):
        self.model = YOLO(model_path)
    
    def detect_and_crop(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, []
        
        results = self.model(image)
        cropped_plates = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = image[y1:y2, x1:x2]
                cropped_plates.append((cropped, (x1, y1, x2, y2)))
                
        return image, cropped_plates