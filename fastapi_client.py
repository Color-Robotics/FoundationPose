import argparse
import glob
import logging
import os
import requests
import time
from typing import List, Optional

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

logging.basicConfig(level=logging.INFO)


# Create the main window
root = tk.Tk()
root.title("Image Viewer")

# Initial setup for the image label
image_label = tk.Label(root)
image_label.pack()


def draw_pose_on_image(rgb_list, pose, mesh_info):
    rgb = np.array(rgb_list, dtype=np.uint8)
    img = Image.fromarray(rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


def encode_image_to_list(image_fn):
    # Encode the image to bytes using np.array().tobytes()
    image = np.array(Image.open(image_fn))
    return image.tolist()


def get_mesh_info(
    session: requests.Session,
    url: str = "http://127.0.0.1:8000/rubikscube/mesh_info",
):
    data = session.get(url)
    logging.info(f"Data: {data.json()}")
    return data.json()


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
    mesh_info = get_mesh_info(session)

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
        draw_pose_on_image(rgb, response.json()["pose"], mesh_info)
        root.update_idletasks()
        root.update()


if __name__ == "__main__":
    main()
