# Automatic Number Plate Recognition System (ANPR)

## Project Overview

This project aims to develop an Automatic Number Plate Recognition (ANPR) system using Python, Computer Vision, and Artificial Intelligence/Machine Learning techniques. The system will detect vehicle number plates from images, video files, and live camera feeds, extract the plate region, and recognize the alphanumeric characters present on the plate.

The system is designed to automate vehicle identification for applications such as parking management, traffic monitoring, security surveillance, toll collection, and access control. In addition to recognizing vehicle registration numbers, the system will maintain logs containing the detected number plate along with the date and time of detection for tracking and record-keeping purposes.

## Workflow

### Phase 1 – Basic Recognition

1. Input an image containing a vehicle.
2. Detect the number plate using computer vision and object detection techniques.
3. Extract and preprocess the detected plate region.
4. Apply OCR or AI-based text recognition to identify the characters.
5. Display the recognized vehicle registration number.

### Phase 2 – Video Processing

1. Accept video input containing moving vehicles.
2. Process video frames in real time.
3. Detect and recognize number plates appearing in the video.
4. Display recognition results for each detected vehicle.

### Phase 3 – Live Monitoring and Logging

1. Connect to a live camera or CCTV stream.
2. Continuously monitor incoming video frames.
3. Detect and recognize vehicle number plates in real time.
4. Record the detected number plate along with the date and timestamp.
5. Store all detections in a database for future search, filtering, and analysis.
6. Display live detection results through a user-friendly interface or dashboard.

## Expected Outcome

The final system will provide an end-to-end solution capable of recognizing vehicle number plates from images, videos, and live camera streams. It will automatically maintain records of detected vehicles with timestamps, enabling efficient vehicle monitoring, tracking, and management in real-world environments.
