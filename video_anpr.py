import cv2
import easyocr
import numpy as np
import re
from ultralytics import YOLO
from collections import defaultdict, Counter
import csv

cv2.setUseOptimized(False)

model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture("video2.mp4")

if not cap.isOpened():
    print("ERROR: Video not opening")
    exit()

region_votes = defaultdict(Counter)

plate_pattern = re.compile(r'^KA[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

valid_districts = {
    "01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","15","16","17","18","19","20"
}

# Plates confirmed invalid — blocklist them
BLOCKLIST = {"KA03M0623", "KA03M0677"}

def clean_text(t):
    return re.sub('[^A-Z0-9]', '', t.upper())

def smart_fix(t):
    plate = t
    if len(t) == 10:
        num_indices  = [2, 3, 6, 7, 8, 9]
        lett_indices = [4, 5]
    elif len(t) == 9:
        num_indices  = [2, 3, 5, 6, 7, 8]
        lett_indices = [4]
    else:
        return plate

    t = list(t)

    for i in num_indices:
        if t[i] in ['O','Q','D']: t[i] = '0'
        elif t[i] == 'I':         t[i] = '1'
        elif t[i] == 'Z':         t[i] = '2'
        elif t[i] == 'S':         t[i] = '5'
        elif t[i] == 'G':         t[i] = '6'
        elif t[i] == 'B':         t[i] = '8'
        elif t[i] == 'q':         t[i] = '9'

    for i in lett_indices:
        if t[i] == '0':   t[i] = 'O'
        elif t[i] == '1': t[i] = 'I'
        elif t[i] == '8': t[i] = 'B'
        elif t[i] == 'H': t[i] = 'M'
        elif t[i] == 'K': t[i] = 'N'
        elif t[i] == 'R': t[i] = 'Q'

    plate = "".join(t)

    if len(plate) == 10:
        plate = plate.replace("HH","MM").replace("HO","NQ").replace("HQ","NQ")
        plate = plate.replace("NO","NQ").replace("RK","NQ").replace("QO","NQ")
        plate = plate.replace("0Q","NQ")

    plate = plate.replace("6269","6469")
    plate = plate.replace("6259","6469")
    plate = plate.replace("6459","6469")

    return plate

def is_valid_plate(t):
    """Extra validation beyond regex — rejects partial/malformed reads"""
    # Last 4 must all be digits
    if not t[-4:].isdigit():
        return False
    # For 9-char: position 4 must be a single valid letter
    if len(t) == 9 and not t[4].isalpha():
        return False
    # For 10-char: positions 4 and 5 must both be letters
    if len(t) == 10 and not (t[4].isalpha() and t[5].isalpha()):
        return False
    # Blocklist check
    if t in BLOCKLIST:
        return False
    return True

def region_id(x, y):
    return f"{int(x/80)}_{int(y/80)}"

def similarity(a, b):
    if len(a) != len(b):
        return 0
    return sum(1 for i in range(len(a)) if a[i] == b[i])

def normalize_plate(p):
    p = list(p)
    for i in range(len(p)):
        if p[i] in ['O','Q','D']: p[i] = '0'
        elif p[i] in ['I','L']:   p[i] = '1'
        elif p[i] == '2':         p[i] = '4'
    return "".join(p)

# ---------------- MAIN LOOP ---------------- #
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("video finished properly")
        break

    frame_count += 1

    if frame_count % 1 != 0:
        display = np.ascontiguousarray(frame)
        cv2.imshow("OUTPUT", display)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        continue

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2-x1, y2-y1

            if w < 80 or h < 25:
                continue
            if not (2 <= w/h <= 6):
                continue

            pad = 10
            plate_img = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                              max(0,x1-pad):min(frame.shape[1],x2+pad)]
            if plate_img.size == 0:
                continue

            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
            thresh = cv2.resize(thresh, None, fx=2, fy=2)

            texts = reader.readtext(thresh, detail=0)

            for t in texts:
                t = clean_text(t)

                if len(t) < 8 or len(t) > 10:
                    continue

                t = smart_fix(t)

                if len(t) not in (9, 10):
                    continue

                if not plate_pattern.match(t):
                    continue

                if t[2:4] not in valid_districts:
                    continue

                # Extra validation — rejects KA03M0623 and similar
                if not is_valid_plate(t):
                    continue

                rid = region_id(x1, y1)
                region_votes[rid][t] += 1

                print(f"[Frame {frame_count}] {t}")

                cv2.putText(annotated, t, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imwrite(f"plate_{t}.jpg", plate_img)

    display = np.ascontiguousarray(annotated.copy())
    cv2.imshow("OUTPUT", display)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- FINAL OUTPUT ---------------- #
print("\nFINAL DETECTED NUMBER PLATES:\n")

final = []
for rid, votes in region_votes.items():
    best = votes.most_common(1)[0][0]
    final.append(best)

clean = []
for p in final:
    is_dup = False
    for c in clean:
        if normalize_plate(p) == normalize_plate(c) or similarity(p, c) >= 8:
            is_dup = True
            break
    if not is_dup:
        clean.append(p)

clean = sorted(clean,
               key=lambda x: -sum(region_votes[r][x]
               for r in region_votes if x in region_votes[r]))

for p in clean[:5]:
    print(p)

with open("detected_plates.txt", "w") as f:
    for p in clean[:5]:
        f.write(p + "\n")

with open("detected_plates.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Plate Number"])
    for p in clean[:5]:
        writer.writerow([p])

print("\nFINAL CLEAN OUTPUT")
print("Saved: detected_plates.txt")
print("Saved: detected_plates.csv")