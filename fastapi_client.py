import argparse
import glob
import os
import base64
import io
import requests
from typing import List

from PIL import Image
import numpy as np


def encode_image_to_list(image_fn):
    # Encode the image to bytes using np.array().tobytes()
    image = np.array(Image.open(image_fn))
    return image.tolist()


def send_image_to_endpoint(rgb: List, depth: List):
    # Assume a tunnel is built to the FastAPI server.
    url = "http://127.0.0.1:8000/rubikscube/inference"
    headers = {"Content-Type": "application/json"}
    data = {"rgb": rgb, "depth": depth}
    response = requests.post(url, json=data, headers=headers)
    return response


def main():
    # Argparse with argument data dir
    parser = argparse.ArgumentParser(description="FastAPI client")
    parser.add_argument("data_dir", type=str, help="Data directory")
    args = parser.parse_args()

    # Send a POST request to the /api/run-pose-estimation endpoint
    for rgb_image in glob.glob(os.path.join(args.data_dir, "rgb", "*.png")):
        # V hacky
        depth_image = rgb_image.replace("rgb", "depth")
        rgb = encode_image_to_list(rgb_image)
        depth = encode_image_to_list(depth_image)
        response = send_image_to_endpoint(rgb, depth)
        # Should get back a pose (3x3 matrix)
        print(f"For image {rgb_image}, pose: {response.json()}")
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    main()
