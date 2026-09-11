from ultralytics import YOLO
import easyocr
import cv2
import os
import re

# OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Folder containing cropped plates
plates_folder = "cropped_plates"

print("Starting OCR on cropped plates...\n")

for file in os.listdir(plates_folder):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(plates_folder, file)

        print("Processing:", file)

        img = cv2.imread(img_path)

        if img is None:
            print("Could not read image")
            continue

        # improve OCR quality
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)

        result = reader.readtext(gray)

        detected = False

        for detection in result:

            text = detection[1]

            # clean unwanted characters
            plate_text = re.sub(r'[^A-Z0-9]', '', text.upper())

            if len(plate_text) >= 5:
                print("Detected Plate:", plate_text)
                detected = True

        if not detected:
            print("OCR Failed")

print("\nOCR finished.")