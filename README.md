# Multi Model Real Time Object Detection AI Camera by Filip Ilovsky

This project is an advanced AI camera system that integrates two YOLOv8 models trained on COCO (80 classes) and OpenImages V7 (600+ classes) to achieve broad spectrum object recognition. Predictions are combined using Weighted Box Fusion (WBF) to intelligently merge overlapping detections, resulting in duplicate free, higher-confidence results beyond the capability of a single model. The pipeline is optimized with OpenCV for real-time video processing, enabling accurate detection across more than 600 object categories.

This project demonstrates expertise in artificial intelligence, machine learning, deep learning model integration, ensemble methods, and real time computer vision systems.

## ✨ Features
- Combines COCO + OpenImages V7 models for broad object coverage  
- Uses Weighted Box Fusion (WBF) for duplicate free, high-confidence detections  
- Runs in real time on webcam or video input  
- Configurable model sizes (s/m/l) and image resolutions to balance speed vs accuracy  

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Filip-2002/AI-live-camera.git
   cd AI-live-camera


2. Create and activate a virtual environment:

   Windows

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   Max/Linux

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt


4. Download YOLOv8 pretrained weights and place them in the project folder:

Default (what this project uses):  
- [YOLOv8l COCO weights (yolov8l.pt)](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)  
- [YOLOv8l OpenImages V7 weights (yolov8l-oiv7.pt)](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt)




Optional (other sizes available for different speed/accuracy trade-offs):  
- [YOLOv8m COCO weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)  
- [YOLOv8m OpenImages V7 weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt)  
- [YOLOv8s COCO weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)  
- [YOLOv8s OpenImages V7 weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt)  

More model sizes can be found on the [Ultralytics YOLOv8 releases page](https://github.com/ultralytics/assets/releases).


5. Run the webcam:
  ```bash
  python run_webcam_wbf.py
  ```

## ⚠️ Notes

- macOS users: Allow your Terminal app access to the Camera in System Settings → Privacy & Security → Camera.

- If you see ModuleNotFoundError (e.g., cv2), make sure your virtual environment is activated (look for (.venv) in your terminal prompt).

- Windows users: If `python` doesn’t work, try using `python3` instead.  

- If you get `pip` version errors, upgrade pip inside the virtual environment (Windows):  
  ```bash
  python -m pip install --upgrade pip

- If you get `pip` version errors, upgrade pip inside the virtual environment (Mac/Linux):  
  ```bash
  python3 -m pip install --upgrade pip

- When switching between projects, deactivate your virtual environment with:
  ```bash
  deactivate

- If you see OpenCV camera errors on macOS, make sure no other application (e.g. Zoom, Teams, or browser) is already using the webcam.

- You can adjust the input image size in `run_webcam_wbf.py` on line 55 ("--imgsz") depending on your machine’s performance:  
  - If your machine is struggling, set the default to **320**.  
  - If your machine is powerful, set the default to **1280**.  
  - The default value (**640**) is a balanced option. 

- You can change the model size in `run_webcam_wbf.py` on lines 52 ("--coco") and 53 ("--oiv7") depending on your machine’s performance:  
  - If your machine is struggling, use `"yolov8s.pt"` and `"yolov8s-oiv7.pt"`, or `"yolov8m.pt"` and `"yolov8m-oiv7.pt"`.  
  - Make sure to download the corresponding model weights in **Step 4**. 
