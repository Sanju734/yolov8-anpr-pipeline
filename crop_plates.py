from ultralytics import YOLO
import cv2
import os

print("Script started...")

model = YOLO("best.pt")

output_folder = "cropped_plates"
os.makedirs(output_folder, exist_ok=True)

extensions = (".jpg", ".jpeg", ".png")

for file in os.listdir():
    
    if file.lower().endswith(extensions):

        print("Processing:", file)

        img = cv2.imread(file)

        if img is None:
            print("Could not read:", file)
            continue

        results = model(img)

        for r in results:

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box)

                plate = img[y1:y2, x1:x2]

                save_path = os.path.join(output_folder, f"{file}_plate_{i}.jpg")

                cv2.imwrite(save_path, plate)

                print("Saved:", save_path)

print("Script finished.")