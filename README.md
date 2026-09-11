Real-Time Automatic Number Plate Recognition (ANPR) Pipeline

An end-to-end computer vision pipeline engineered to detect, extract, and log vehicle license plate numbers from live video streams and image feeds.

Key Features
- Custom YOLOv8 object detection model fine-tuned for vehicle license plate localization.
- OpenCV image preprocessing pipeline implementing grayscale conversion, Gaussian blur, and adaptive thresholding.
- EasyOCR engine integrated with regex filtering matching standard Indian vehicle registration formats.
- Multi-frame temporal voting mechanism to eliminate character flicker and false-positive readings across sequential video frames.
- Automated data logging pipeline saving verified plate records directly to CSV and text files.

 Tech Stack
- Programming Language: Python
- Deep Learning & Computer Vision: PyTorch, Ultralytics YOLOv8, OpenCV, EasyOCR
- Data Processing: NumPy, Pandas, Regex

Repository Structure
- `final_anpr.py`: Main inference script executing plate detection and OCR.
- `video_anpr.py`: Video stream processing pipeline with real-time bounding box overlays.
- `ocr_plate.py`: Image preprocessing and EasyOCR text extraction module.
- `crop_plates.py`: Bounding box cropping utility for detected plate regions.
- `best.pt`: Fine-tuned YOLOv8 model weights.
- `detected_plates.csv`: Structured output log containing verified plate numbers and timestamps.
