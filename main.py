import cv2
from yolo import YOLO
import wave
import time
from utils import (
    filter_signal,
    display_hands,
    filter_signal_butter,
    get_cutoff,
    apply_effect,
    draw_axis,
)
import pyaudio
import numpy as np


from args_control import get_configs


args = get_configs()
print("loading yolo...")
if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO(
        "models/cross-hands-tiny-prn.cfg",
        "models/cross-hands-tiny-prn.weights",
        ["hand"],
    )
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO(
        "models/cross-hands-yolov4-tiny.cfg",
        "models/cross-hands-yolov4-tiny.weights",
        ["hand"],
    )
else:
    print("loading yolo-tiny...")
    yolo = YOLO(
        "models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"]
    )

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)
filter_type = args.filter
cutoff_type = args.cutoff
effect_type = args.effect

max_freq = args.max_freq
max_depth = args.max_depth
min_freq = args.min_freq
min_depth = args.min_depth
duration = args.duration

start = time.time()


def callback(in_data, frame_count, time_info, status):

    fs = wf.getframerate()
    data = wf.readframes(frame_count)

    if len(data) % 2 != 0:
        data += b"\x00"

    if cutoff:
        int_data = np.frombuffer(data, dtype=np.int16)
        final_data = filter_signal(int_data, fs, cutoff, filter_type)
        # filtered_data = filter_signal_butter(int_data, fs, cutoff, filter_type)
        if effect_type:
            final_data = apply_effect(final_data, effect_type, depth, fs, cx)

        final_data = np.clip(final_data, -32768, 32767)
        final_data = final_data.astype(np.int16)

        data = final_data.tobytes()
    return (data, pyaudio.paContinue)


print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(args.device)
width = vc.get(3)  # float `width`
height = vc.get(4)


# print("web cam width, and height", width, height)
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False


width, height, inference_time, results = yolo.inference(frame)
hand_count = len(results)

if args.hands != -1:
    hand_count = int(args.hands)

print("starting audio...")

filename = f"audios/{args.filename}.wav"
cutoff, depth, frame_count = 500, 0, 16384
wf = wave.open(filename, "rb")

p = pyaudio.PyAudio()


stream = p.open(
    format=p.get_format_from_width(wf.getsampwidth()),
    channels=wf.getnchannels(),
    rate=wf.getframerate(),
    output=True,
    stream_callback=callback,
    frames_per_buffer=frame_count,
)

# Start the stream
stream.start_stream()

#  Keep the stream active
try:
    # prev = 0
    while stream.is_active() and (time.time() - start) < duration:
        rval, frame = vc.read()

        draw_axis(frame, filter_type, effect_type)
        # Display the resulting frame
        width, height, inference_time, results = yolo.inference(frame)

        cx, cy = display_hands(results, frame, hand_count, inference_time)
        # sort by confidence
        results.sort(key=lambda x: x[2])

        # how many hands should be shown
        hand_count = len(results)
        if args.hands != -1:
            hand_count = int(args.hands)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

        cutoff = get_cutoff(cx, min_freq, max_freq, width, cutoff_type=cutoff_type)
        depth = min_depth + 1 - cy / height
        print(cutoff, depth)

except KeyboardInterrupt:
    print("Keyboard interrupt received. Exiting gracefully.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop the stream
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()

    # stop the webcam
    cv2.destroyWindow("preview")
    vc.release()
