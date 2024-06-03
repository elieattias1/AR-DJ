# VR-DJ

This project combines hand detection using the YOLO (You Only Look Once) algorithm and real-time audio processing. The application captures video from a webcam, detects hands in each frame, and processes audio data based on the detected hand positions.

## Features

- **Hand Detection**: Uses YOLO to detect hands in real-time.
- **Audio Processing**: Applies filters to audio signals based on hand positions.
- **Real-Time Display**: Shows the webcam feed with detected hand bounding boxes and plays processed audio.

## Install Dependencies
`pip install -r requirements.txt
`
## Download Models
```bash
# mac / linux
cd models && sh ./download-models.sh

# windows
cd models && powershell .\download-models.ps1

```

## Run the Application 
```bash
# with python 3
python demo_webcam.py
```

### Models

- YOLOv3 Cross-Dataset
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights)
- YOLOv3-tiny Cross-Hands
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.weights)
- YOLOv3-tiny-prn Cross-Hands
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.weights)
- YOLOv4-Tiny Cross-Hands
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-yolov4-tiny.weights)



