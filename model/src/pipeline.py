import argparse
import cv2
from detect import PlateDetector
from preprocess import clean_plate_image
from ocr import extract_text

def main(image_path, model_path):
    detector = PlateDetector(model_path=model_path)
    original_image, plates = detector.detect_and_crop(image_path)
    
    if not plates:
        print("No plates detected.")
        return

    for idx, (cropped_plate, coords) in enumerate(plates):
        cleaned_plate = clean_plate_image(cropped_plate)
        text = extract_text(cleaned_plate)
        
        print(f"Plate {idx + 1}: {text}")
        
        x1, y1, x2, y2 = coords
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Detected Plates", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="weights/yolov8n.pt")
    args = parser.parse_args()
    
    main(args.image, args.model)