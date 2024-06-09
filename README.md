# Reflective-Reality-DJ

This project combines the immersive experience of AR with hand detection using YOLO (You Only Look Once) and real-time audio processing. The application captures video from a webcam, detects hand positions in each frame, and transforms audio based on these positions, creating an interactive and immersive VR DJ experience. The aim of this project is to pioneer a new wave of interactivity in DJing, setting the stage for a future where DJs not only play music but also dance alongside their audience, creating a more engaging and dynamic performance experience.

## Features

- **Hand Detection**: Utilizes YOLO for precise real-time hand tracking.
- **Interactive Real Time Audio Processing**: Modulates audio effects dynamically based on hand movements.
- **Immersive Experience**: Displays a live webcam feed with hand tracking overlays and synchronized audio effects, enhancing the sense of immersion.
- **Audio Filters**: Supports low-pass and high-pass filters based on hand movement along the x-axis.
- **Audio Effects**: Applies tremolo or vibrato effects based on hand movement along the y-axis.



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
Add your audio files to the audios folder (as .wav files) in the project directory. 

Run the application with the default low-pass filter:

```bash
# with python 3
python main.py --filename=your-audio
```

Apply a low-pass filter (hand moving along the x-axis) with varying cut-off frequency:
```bash
# with python 3
python main.py --filename=your-audio --filter=low
```

Apply a high-pass filter (hand moving along the x-axis) with varying cut-off frequency:

```bash
# with python 3
python main.py --filename=your-audio --filter=high
```

Apply post-filtering sound effects (hand moving along the y-axis) with varying depth:
```bash
# with python 3
python main.py --filename=your-audio --effect=tremolo
```
```bash
# with python 3
python main.py --filename=your-audio --effect=vibrato
```

### Models

The following models trained by [cansik](https://github.com/cansik) can be selected by adjusting the --network parameter: `normal`, `tiny`, `prn`, and `v4-tiny`.

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

### Considerations
Larger models result in longer inference times, reducing the number of frames processed per second. Conversely, smaller models offer better performance and a smoother experience, though they may compromise inference quality.

## Acknowledgments

This project is built on top of the [cansik's](https://github.com/cansik) contribution within the [YOLO-Hand-Detection](https://github.com/cansik/yolo-hand-detection/tree/master) project.

