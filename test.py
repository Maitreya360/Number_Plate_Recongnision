import cv2
import re
from collections import defaultdict, deque, Counter
from ultralytics import YOLO
import easyocr
from difflib import get_close_matches

# Load model and OCR
model = YOLO("license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# Indian plate format
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$")


VALID_STATE_CODES = [
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JK","JH",
    "KA","KL","MP","MH","MN","ML","MZ","NL","OR","PB","RJ",
    "SK","TN","TR","UK","UP","WB","AN","CH","DH","DD","DL",
    "LD","PY"
]

def fix_state_code(plate):
    if len(plate) < 2:
        return plate

    state = plate[:2]

    if state in VALID_STATE_CODES:
        return plate

    match = get_close_matches(
        state,
        VALID_STATE_CODES,
        n=1,
        cutoff=0.5
    )

    if match:
        return match[0] + plate[2:]

    return plate

def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "6": "G",
    "7": "T",
    "2": "Z"
}

    mapping_alpha_to_num = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    ocr_text = ''.join(ch for ch in ocr_text.upper() if ch.isalnum())

    if len(ocr_text) != 10:
        return ""

    corrected = []

    for i, ch in enumerate(ocr_text):

        # State letters
        if i in [0, 1]:
            if ch.isdigit():
                corrected.append(mapping_num_to_alpha.get(ch, ch))
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""

        # District digits
        elif i in [2, 3]:
            if ch.isalpha():
                corrected.append(mapping_alpha_to_num.get(ch, ch))
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

        # Series letters
        elif i in [4, 5]:
            if ch.isdigit():
                corrected.append(mapping_num_to_alpha.get(ch, ch))
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""

        # Vehicle number digits
        else:
            if ch.isalpha():
                corrected.append(mapping_alpha_to_num.get(ch, ch))
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    plate = ''.join(corrected)

    # Fix state code to closest valid Indian code
    plate = fix_state_code(plate)
    
    if plate_pattern.match(plate):
        return plate
    
    return ""


def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    plate_resized = cv2.resize(
        thresh,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0])

            if candidate and plate_pattern.match(candidate):
                return candidate

    except Exception:
        pass

    return ""


# Plate stabilization
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}
all_plates = []
first_seen = {}
last_seen = {}

def get_box_id(x1, y1, x2, y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"


def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)

        most_common = max(
            set(plate_history[box_id]),
            key=plate_history[box_id].count
        )

        plate_final[box_id] = most_common

    return plate_final.get(box_id, "")


def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:05.2f}"


# Video paths
input_video = "video.mp4"
output_video = "output_with_licensev3.mp4"

cap = cv2.VideoCapture(input_video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_video,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

CONF_THRESH = 0.30

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:

            conf = float(box.conf.cpu().item())

            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy.cpu().numpy()[0]
            )

            plate_crop = frame[y1:y2, x1:x2]

            # OCR
            text = recognize_plate(plate_crop)

            # Stabilize OCR
            box_id = get_box_id(x1, y1, x2, y2)
            stable_text = get_stable_plate(box_id, text)
            if stable_text:

                frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps > 0:
                    timestamp = frame_no / fps

                    all_plates.append(stable_text)

                    if stable_text not in first_seen:
                        first_seen[stable_text] = timestamp

                    last_seen[stable_text] = timestamp
            

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Show zoomed plate
            if plate_crop.size > 0:

                overlay_h = 150
                overlay_w = 400

                plate_zoom = cv2.resize(
                    plate_crop,
                    (overlay_w, overlay_h)
                )

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1

                oy2 = oy1 + overlay_h
                ox2 = ox1 + overlay_w

                if (
                    oy2 <= frame.shape[0]
                    and ox2 <= frame.shape[1]
                ):

                    frame[oy1:oy2, ox1:ox2] = plate_zoom

                    if stable_text:

                        cv2.putText(
                            frame,
                            stable_text,
                            (ox1, oy1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 0),
                            6
                        )

                        cv2.putText(
                            frame,
                            stable_text,
                            (ox1, oy1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 255, 255),
                            3
                        )

            print("Detected:", stable_text)

    out.write(frame)

if len(all_plates) > 0:

    counts = Counter(all_plates)

    best_plate = counts.most_common(1)[0][0]
    occurrences = counts[best_plate]

    confidence = occurrences / len(all_plates)

    with open("plate_results.txt", "w") as f:
        f.write(f"Plate: {best_plate}\n")
        f.write(f"Confidence: {confidence:.2f}\n")
        f.write(
            f"First Seen: {format_time(first_seen[best_plate])}\n"
        )
        f.write(
            f"Last Seen: {format_time(last_seen[best_plate])}\n"
        )
        f.write(
            f"Occurrences: {occurrences}/{len(all_plates)}\n"
        )

    print("\n===== FINAL RESULT =====")
    print("Plate:", best_plate)
    print("Confidence:", round(confidence, 2))
    print("Saved to plate_results.txt")

else:
    print("No valid plate detected.")

cap.release()
out.release()

print("✔ Annotated video saved as:", output_video)