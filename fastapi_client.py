import argparse
import glob
import logging
import os
import requests
import time
from typing import List, Optional

from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)


def encode_image_to_list(image_fn):
    # Encode the image to bytes using np.array().tobytes()
    image = np.array(Image.open(image_fn))
    return image.tolist()


def send_image_to_endpoint(
    rgb: List,
    depth: List,
    session: Optional[requests.Session] = None,
):
    # Assume a tunnel is built to the FastAPI server.
    url = "http://127.0.0.1:8000/rubikscube/inference"
    headers = {"Content-Type": "application/json"}
    data = {"rgb": rgb, "depth": depth}
    if session:
        response = session.post(url, json=data, headers=headers)
        logging.info("using the existing session")
    else:
        response = requests.post(url, json=data, headers=headers)
    return response


def main():
    # Argparse with argument data dir
    parser = argparse.ArgumentParser(description="FastAPI client")
    parser.add_argument("data_dir", type=str, help="Data directory")
    args = parser.parse_args()

    session = requests.Session()
    session.get("http://127.0.0.1:8000/ping")

    # Send a POST request to the /api/run-pose-estimation endpoint
    for rgb_image in glob.glob(os.path.join(args.data_dir, "rgb", "*.png")):
        # V hacky
        depth_image = rgb_image.replace("rgb", "depth")
        rgb = encode_image_to_list(rgb_image)
        depth = encode_image_to_list(depth_image)
        send_time = time.time()
        response = send_image_to_endpoint(rgb, depth, session)
        latency = time.time() - send_time
        # Should get back a pose (3x3 matrix)
        print(f"[{latency:.3f}s] For image {rgb_image}, pose: {response.json()}")
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    main()
