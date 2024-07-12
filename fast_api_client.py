import argparse
import glob
import os
import base64
import io
import requests

from PIL import Image
import numpy as np


def encode_image_to_bytes(image_fn):
    # Encode the image to bytes using np.array().tobytes()
    image = np.array(Image.open(image_fn))
    return image.tobytes()


def send_image_to_endpoint(rgb: np.ndarray, depth: np.ndarray):
    # Assume a tunnel is built to the FastAPI server.
    url = "http://127.0.0.1:8000/rubikscube/inference"
    headers = {"Content-Type": "application/json"}
    data = {"image_base64": rgb.tolist(), "depth_base64": depth.tolsit()}
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
        import pdb

        pdb.set_trace()
        rgb = encode_image_to_bytes(rgb_image)
        depth = encode_image_to_bytes(depth_image)
        response = send_image_to_endpoint(rgb, depth)
        # Should get back a pose (3x3 matrix)
        print(f"For image {rgb_image}, pose: {response.json()}")


if __name__ == "__main__":
    main()
