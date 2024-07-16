import base64

import fastapi
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
import pydantic
import numpy as np
import logging

from run_color_demo import FoundationPoseStream
import run_color_demo
import time

CODE_DIR = run_color_demo.CODE_DIR

# Using fastapi create a web server
# There should be a route for each object to detect.
# Would be cool if you could post to setup the route and then get back the new route to post the images to.
# If you post an image, you get back the pose of the object if it exists.
# Add an arg to say "reset" the tracker so that we start pushing things through an mask model again.
app = fastapi.FastAPI()


# TODO: How to handle different objects easily?  CLI args?
# TODO: Fix this path
RUBIKS_CUBE_DETECTOR = FoundationPoseStream(
    f"{CODE_DIR}/../data/rubikscube/mesh/rubiks_cube_scaled.obj",
    f"{CODE_DIR}/../data/rubikscube/cam_K.txt",
    f"{CODE_DIR}/../data/rubikscube/masks/1712769756892.png",  # HACK!
)


class RGBDData(pydantic.BaseModel):
    """Message schema for sending data for inference."""

    rgb: list
    depth: list
    rgb_shape: list
    depth_shape: list


@app.get("/rubikscube/mesh_info")
def mesh_info():
    try:
        return {
            "to_origin": RUBIKS_CUBE_DETECTOR.to_origin.tolist(),
            "bbox": RUBIKS_CUBE_DETECTOR.bbox.tolist(),
            "k": RUBIKS_CUBE_DETECTOR.K.tolist(),
        }
    except Exception as e:
        logging.exception("Failed to get mesh info")
        raise HTTPException(
            status_code=400, detail=f"Failed to get mesh info: {str(e)}"
        )


@app.post("/rubikscube/inference")
def pose_inference(data: RGBDData):
    try:
        start_time = time.time()
        print(f"inference time: {start_time:.3f}")

        # Use the provided shape information
        rgb_shape = tuple(data.rgb_shape)
        depth_shape = tuple(data.depth_shape)

        image_data = np.frombuffer(bytes(data.rgb), dtype=np.uint8).reshape(rgb_shape)
        depth_data = np.frombuffer(bytes(data.depth), dtype=np.uint8).reshape(depth_shape)
        send_time = time.time()
        print(f"post array sizing time: {send_time:.3f}")
        latency = send_time - start_time
        print(f"[post array sizing latency: {latency:.3f}s]")

        pose = RUBIKS_CUBE_DETECTOR.detect(image_data, depth_data)

        latency = time.time() - send_time
        print(f"[pose latency: {latency:.3f}s]")
        start_time = time.time()
        print(f"pose return time: {start_time:.3f}s")
        return {"pose": pose.tolist()}
    except Exception as e:

        logging.exception("Failure")
        raise HTTPException(
            status_code=400, detail=f"Failed to process capture: {str(e)}"
        )


@app.get("/ping")
def ping():
    return {"message": "pong"}
