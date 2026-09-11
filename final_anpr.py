from ultralytics import YOLO
import easyocr
import cv2
import os
import re
import csv

# Load YOLO model
model = YOLO("best.pt")

# OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Output folder
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# CSV file
csv_file = open("plate_results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image", "Plate Number"])

# Image extensions
extensions = (".jpg", ".jpeg", ".png")

print("Starting Automatic Number Plate Recognition...\n")

for file in os.listdir():

    if file.lower().endswith(extensions):

        print("Processing:", file)

        img = cv2.imread(file)

        if img is None:
            print("Could not read image")
            continue

        results = model(img)

        for r in results:

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                plate = img[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                # Improve OCR
                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2)

                result = reader.readtext(gray)

                for detection in result:

                    text = detection[1]

                    # Clean text
                    plate_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                    if len(plate_text) >= 5:

                        print("Detected Plate:", plate_text)

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

                        # Write plate text
                        cv2.putText(img, plate_text,
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0,255,0),
                                    2)

                        csv_writer.writerow([file, plate_text])

        # Save output image
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, img)

print("\nProcessing finished.")

csv_file.close()