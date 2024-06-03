import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import cv2


def filter_signal_butter(y, sr, cutoff, filter_type):
    sos = signal.butter(5, cutoff, filter_type, fs=sr, output="sos")
    filtered_signal = signal.sosfilt(sos, y)
    return filtered_signal


def weighting_function(x):

    return 2 / (1 + np.exp(0.01 * (x) ** 2))


def apply_tremolo(y, sr, depth=0.5, freq=5.0):
    """Apply tremolo effect to a signal."""
    t = np.arange(len(y)) / sr
    modulator = 1.0 + depth * np.sin(2 * np.pi * freq * t)
    return y * modulator


def apply_vibrato(y, sr, depth=0.005, freq=5.0):
    """Apply vibrato effect to a signal."""
    t = np.arange(len(y)) / sr
    modulator = np.sin(2 * np.pi * freq * t) * depth
    indices = np.arange(len(y)) + modulator * sr
    indices = np.clip(indices, 0, len(y) - 1).astype(np.int16)
    return y[indices]


def filter_signal(y, sr, cutoff, filter_type, effect="tremolo"):
    """Filter signal using FFT."""

    freqs = np.fft.fftfreq(len(y), d=1 / sr)
    fft = np.fft.fft(y)

    shifted_freqs = np.fft.fftshift(freqs)
    n = len(shifted_freqs)
    xx = np.linspace(-n // 2, n // 2, n)

    if filter_type == "low":
        # weights = weighting_function(xx)
        # print(weights)
        fft[np.abs(freqs) > cutoff] = 0

    elif filter_type == "high":
        # weights = 1 - weighting_function(xx)
        fft[np.abs(freqs) < cutoff] = 0

    y_filtered = np.fft.ifft(fft).real

    return y_filtered


def apply_effect(y, effect_type, depth, sr):
    if effect_type == "tremolo":
        y = apply_tremolo(y, depth=depth, sr=sr)
    elif effect_type == "vibrato":
        y = apply_vibrato(y, depth=depth / 1000, sr=sr)
    return y


def get_cutoff(cx, min_freq, max_freq, width, cutoff_type="linear"):
    if cutoff_type == "linear":
        return min_freq + cx * (max_freq / width)
    elif cutoff_type == "quadratic":
        print("cx", cx)
        return min_freq + max_freq * cx**2 * (1 / width) ** 2


def display_hands(results, frame, hand_count, inference_time):
    cv2.putText(
        frame,
        f"{round(1/inference_time,2)} FPS",
        (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    cx = None
    cy = None
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)
        # print(cx, cy)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("preview", frame)
    if cx is None or cy is None:
        return 0, 0
    return cx, cy


def write_wav(y, sr, savepath):
    wavfile.write(savepath, sr, y.astype(np.float32))


def plot_spectrograms(original_signal, filtered_signal, sr):
    """Plot spectrograms of original and filtered signals."""
    fig, axes = plt.subplots(2, 1, figsize=(19, 5))

    # Plot spectrogram for the original signal
    D_original = librosa.amplitude_to_db(
        np.abs(librosa.stft(original_signal)), ref=np.max
    )
    img1 = librosa.display.specshow(
        D_original, sr=sr, x_axis="time", y_axis="log", ax=axes[0]
    )
    axes[0].set_title("Original Signal")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # Plot spectrogram for the filtered signal
    D_filtered = librosa.amplitude_to_db(
        np.abs(librosa.stft(filtered_signal)), ref=np.max
    )
    img2 = librosa.display.specshow(
        D_filtered, sr=sr, x_axis="time", y_axis="log", ax=axes[1]
    )
    axes[1].set_title("Filtered Signal")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()
