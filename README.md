# Multi-Model Real-Time Object Detection AI Camera by Filip Ilovsky

This project is an advanced AI camera system that integrates two YOLOv8 models trained on COCO (80 classes) and OpenImages V7 (600+ classes) to achieve broad-spectrum object recognition. Predictions are combined using Weighted Box Fusion (WBF) to intelligently merge overlapping detections, resulting in duplicate-free, higher-confidence results beyond the capability of a single model. The pipeline is optimized with OpenCV for real-time video processing, enabling accurate detection across more than 600 object categories.

This project demonstrates expertise in artificial intelligence, machine learning, deep learning model integration, ensemble methods, and real-time computer vision systems.

## ✨ Features
- Combines COCO + OpenImages V7 models for broad object coverage  
- Uses Weighted Box Fusion (WBF) for duplicate-free, high-confidence detections  
- Runs in real time on webcam or video input  
- Configurable model sizes (s/m/l) and image resolutions to balance speed vs accuracy  

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Filip-2002/AI-live-camera.git
   cd AI-live-camera


2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # Linux/Mac


3. Install dependencies:
   ```bash
   pip install -r requirements.txt


4. Download YOLOv8 pretrained weights and place them in the project folder:

Default (what this project uses):  
- [YOLOv8l COCO weights (yolov8l.pt)](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)  
- [YOLOv8l OpenImages V7 weights (yolov8l-oiv7.pt)](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt)

Optional (other sizes available for different speed/accuracy trade-offs):  
- [YOLOv8s COCO weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)  
- [YOLOv8s OpenImages V7 weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt)  

More model sizes (`n`, `s`, `m`, `l`, `x`) can be found on the [Ultralytics YOLOv8 releases page](https://github.com/ultralytics/assets/releases).


5. Run the webcam:
  ```bash
  python run_webcam_wbf.py



if your machine isnt as good change the line where the comment is in run_webcam_wbf.py (52 and 53)

if machine is struggling change line 55 to 320 or if machine is very good change to 1280





