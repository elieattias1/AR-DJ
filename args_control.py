import argparse, os


def get_configs():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-n",
        "--network",
        default="normal",
        choices=["normal", "tiny", "prn", "v4-tiny"],
        help="Network Type",
    )
    audios_folder = "audios"
    if not os.path.exists(audios_folder):
        os.makedirs(audios_folder)
        print(f"The folder '{audios_folder}' has been created.")
    else:
        print(f"The folder '{audios_folder}' already exists.")

    ap.add_argument("--filename", type=str, default="", help="Audio file to play")
    ap.add_argument("-d", "--device", type=int, default=0, help="Device to use")

    ap.add_argument("-s", "--size", default=416, help="Size for yolo")
    ap.add_argument("-c", "--confidence", default=0.2, help="Confidence for yolo")
    ap.add_argument(
        "-nh",
        "--hands",
        default=-1,
        help="Total number of hands to be detected per frame (-1 for all)",
    )
    ap.add_argument(
        "--filter",
        default="low",
        type=str,
        help="Low or High pass filter",
    )
    ap.add_argument(
        "--cutoff",
        default="linear",
        type=str,
        help="How cutoff frequency changes",
    )
    ap.add_argument(
        "--effect",
        default="",
        type=str,
        help="Applies effect on top of filtered signal",
    )
    ap.add_argument(
        "--max_freq",
        default=5000,
        type=int,
        help="Maximum cutoff frequency",
    )
    ap.add_argument(
        "--max_depth",
        default=0.5,
        type=int,
        help="Maximum depth of audio effects",
    )
    ap.add_argument(
        "--min_freq",
        default=0,
        type=int,
        help="Minimum cutoff frequency",
    )
    ap.add_argument(
        "--min_depth",
        default=0,
        type=int,
        help="Minimum depth of audio effects",
    )

    ap.add_argument(
        "--duration",
        default=120,
        type=int,
        help="how long the song will play (in seconds)",
    )
    args = ap.parse_args()

    return args


if __name__ == "__main__":
    get_configs()
