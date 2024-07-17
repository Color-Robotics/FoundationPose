import argparse
import glob
import logging
import os
import requests
import time
from typing import List, Optional

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

from dotenv import load_dotenv

load_dotenv()
import sys

logging.basicConfig(level=logging.INFO)


def load_image(image_fn):
    return np.array(Image.open(image_fn))


def encode_image_to_list(image_arr):
    return image_arr.tolist()


def encode_image_to_bytes(image_fn):
    image = np.array(Image.open(image_fn), dtype=np.uint8)
    return image.tobytes(), ",".join(map(str, image.shape))


def get_mesh_info(
    session: requests.Session,
    url: str = "http://127.0.0.1:8000/rubikscube/mesh_info",
):
    data = session.get(url)
    logging.info(f"Data: {data.json()}")
    return data.json()


def send_image_to_endpoint(
    rgb: bytes,
    rgb_shape: str,
    depth: bytes,
    depth_shape: str,
    session: Optional[requests.Session] = None,
):
    # Assume a tunnel is built to the FastAPI server.
    url = "http://127.0.0.1:8000/rubikscube/inference"
    headers = {"Content-Type": "application/json"}
    files = {
        "rgb": ("rgb.bin", rgb, "application/octet-stream"),
        "depth": ("depth.bin", depth, "application/octet-stream"),
    }
    data = {"rgb_shape": rgb_shape, "depth_shape": depth_shape}
    if session:
        response = session.post(url, files=files, data=data)
    else:
        response = requests.post(url, files=files, data=data)
    return response


def main():
    # Argparse with argument data dir
    parser = argparse.ArgumentParser(description="FastAPI client")
    parser.add_argument("data_dir", type=str, help="Data directory")
    args = parser.parse_args()

    session = requests.Session()
    session.get("http://127.0.0.1:8000/ping")
    mesh_info = get_mesh_info(session)

    # Send a POST request to the /api/run-pose-estimation endpoint
    files = glob.glob(os.path.join(args.data_dir, "rgb", "*.png"))
    files.sort()
    for rgb_image in files:
        # V hacky
        depth_image = rgb_image.replace("rgb", "depth")
        rgb, rgb_shape = encode_image_to_bytes(rgb_image)
        depth, depth_shape = encode_image_to_bytes(depth_image)
        send_time = time.time()
        response = send_image_to_endpoint(rgb, rgb_shape, depth, depth_shape, session)
        latency = time.time() - send_time
        # Should get back a pose (3x3 matrix)
        print(f"[{latency:.3f}s] For image {rgb_image}, pose: {response.json()}")
        if not response.ok:
            logging.error(f"Failed to get pose for {rgb_image}")
            continue


if __name__ == "__main__":
    main()
