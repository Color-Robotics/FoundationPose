import base64

import fastapi
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
import pydantic

from run_color_demo import FoundationPoseStream
import run_color_demo

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
)


class ImageData(pydantic.BaseModel):
    """Message schema for sending data for inference."""

    image_base64: str
    depth_base64: str


@app.post("/rubikscube/inference")
def pose_inference(data: ImageData):
    try:
        # Decode the base64 string to bytes
        image_data = base64.b64decode(data.image_base64)
        depth_data = base64.b64decode(data.depth_base64)
        # process the image
        RUBIKS_CUBE_DETECTOR.detect(image_data, depth_data)

        return {"message": "Image processed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process capture: {str(e)}"
        )
