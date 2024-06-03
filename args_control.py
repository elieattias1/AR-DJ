import argparse


def get_configs():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-n",
        "--network",
        default="normal",
        choices=["normal", "tiny", "prn", "v4-tiny"],
        help="Network Type",
    )
    ap.add_argument("-d", "--device", type=int, default=0, help="Device to use")
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
        default="tremolo",
        type=str,
        help="Applies effect on top of filtered signal",
    )
    ap.add_argument("-s", "--size", default=416, help="Size for yolo")
    ap.add_argument("-c", "--confidence", default=0.2, help="Confidence for yolo")
    ap.add_argument(
        "-nh",
        "--hands",
        default=-1,
        help="Total number of hands to be detected per frame (-1 for all)",
    )
    args = ap.parse_args()

    return args


if __name__ == "__main__":
    get_configs()
