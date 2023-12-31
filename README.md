# Object-tracking-and-counting-using-YOLOV

#### Video and Webcam Demo
<div align="center">
<p>
<img src="assets/traffic.gif" width="300"/>  <img src="assets/output_1696816969211397.gif" width="300"/> 
</p>
</div>
<div align="center">
<p>
<img src="assets/output_1696838504179469.gif" width="300"/> 
</p>
</div>


## Introduction

This repository contains the code for object detection, tracking, and counting using the YOLOv8 algorithm by ultralytics for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for object tracking. The project provides code for both procedural and object-oriented programming implementations in Python.

### Features:
* **Web App and API**: Object detection Flask UI and API
* **Object detection**: The YOLOv5 or 7 algorithm has been used to detect objects in images and videos. The algorithm is known for its fast and accurate performance.
* **Object tracking**: The SORT algorithm has been used for tracking the detected objects in real-time. SORT is a simple algorithm that performs well in real-time tracking scenarios.
* **Object counting**: The project also includes a module for counting the number of objects detected in a given image or video.
* **Image Object Detction API**: The project has been implemented using object-oriented programming principles, making it modular and easy to understand.

## Setup
  ```code
  conda create -n <env_name> python==3.10
  conda activate <env_name>
  git clone https://github.com/USTAADCOM/Object-tracking-and-counting-using-YOLOV.git
  cd Object-tracking-and-counting-using-YOLOV
  pip install -r requirements.txt -q
  ```
**Note** Make sure that all dependencies installed, including torch>=1.7.
## Downloaad Model
* Download [yolov7.pt](https://github.com/USTAADCOM/Object-tracking-and-counting-using-YOLOV/releases/download/v1.1/yolov7.pt)
from releases
* Put in weights folder
## Run Flask Web App
```code
python3 app.py 
```
# Object-tracking-and-counting-using-YOLOV API
## Setup
create .env and put your key
```
api-key = "your sceret key here"
```
## Run API
```code
python3 server.py 
```
### http://127.0.0.1:5000/detect_image
Payload
```code
{
"key": "data_file",
"type": "file",		
"src": "image file path"
}
```
Response 
```code
{
    "output": {
        "image_file": "Cloud storage path"
    }
}
```
#### Demo
<div align="center">
<p>
<img src="assets/stock.jpg" width="300"/>  <img src="assets/1696923383705738.jpg" width="300"/> 
</p>
</div>


###  http://127.0.0.1:5000/track_video 
Payload
```code
{
    "key": "data_file",
    "type": "file",
    "src": "Video file path"
}
```
Response 
```code
{
    "output": {
        "video_file": "Cloud storage path"
    }
}
```
###  http://127.0.0.1:5000/image_prediction
Payload
```code
{
    "key": "data_file",
    "type": "file",
    "src": "image file path"
}
```
Response 
```code
{
    "ouptut": {
        "class 1": count,
        "class 2": count,
        "class 3": count,
        .
        .
        "class n": count,    
    }
}