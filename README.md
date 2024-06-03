# VR-DJ

This project combines the immersive experience of VR with hand detection using the YOLO (You Only Look Once) algorithm and real-time audio processing. The application captures video from a webcam, detects hand positions in each frame, and transforms audio based on these positions, creating an interactive and immersive VR DJ experience.

## Features

- **Hand Detection**: Utilizes YOLO for precise real-time hand tracking.
- **Interactive Audio Processing**: Modulates audio effects dynamically based on hand movements.
- **Immersive Experience**: Displays a live webcam feed with hand tracking overlays and synchronized audio effects, enhancing the sense of immersion.


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
python main.py
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

## Acknowledgments

This project is built on top of the cansik's contribution within the [YOLO Hand Detection project](https://github.com/cansik/yolo-hand-detection/tree/master).

