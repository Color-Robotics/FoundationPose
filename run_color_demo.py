# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
import logging
import base64
import glob
from typing import Optional

import trimesh
import numpy as np
import nvdiffrast.torch as dr
import cv2
import imageio
import open3d as o3d
import fastapi
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
import pydantic


import estimater
from learning.training import predict_score
from learning.training import predict_pose_refine
import datareader
import Utils


# Using fastapi create a web server
# There should be a route for each object to detect.
# Would be cool if you could post to setup the route and then get back the new route to post the images to.
# If you post an image, you get back the pose of the object if it exists.
# Add an arg to say "reset" the tracker so that we start pushing things through an mask model again.
app = fastapi.FastAPI()

CODE_DIR = os.path.dirname(os.path.realpath(__file__))


class ImageData(pydantic.BaseModel):
    """Message schema for sending data for inference."""

    image_base64: str
    depth_base64: str


class FoundationPoseStream:
    """Class to handle pose estimation for a stream of images."""

    def __init__(self, mesh_file: str, cam_k_file: str, mask_0: Optional[np.ndarray] = None):
        """Initialize the pose estimation pipeline."""
        self.mesh_file = mesh_file
        self.K = np.loadtxt(cam_k_file).reshape(3, 3)
        self.mask_0 = mask_0
        debug_dir = f"{CODE_DIR}/debug"
        debug = 1
        # TODO: Do this better.
        os.system(
            f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
        )
        Utils.set_seed(0)

        # This stuff should probably be in a DB for faster access.
        mesh = trimesh.load(mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        scorer = predict_score.ScorePredictor()
        refiner = predict_pose_refine.PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = estimater.FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=glctx,
        )
        logging.info("estimator initialization done")

        self.image_counter = 0
        # Flag to determine if we have a mask used for tracking.
        self.has_mask = False
        # I am going to hardcode this.  I don't know what impact decreasing/increasing resolution would have on results?
        # If the results don't change much, might make sense to always resize to the same dimensions.
        self.W = 640
        self.H = 720

        # This is the default, but could make sense to clip it depending on use case.
        self.zfar = np.inf

        # Default from demo
        self.est_refine_iter = 5
        self.track_refine_iter = 2

    def process_color_data(self, color: np.ndarray):
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def process_depth_data(self, depth: np.ndarray):
        """Process the depth data.

        Note: Depth should be in mm.
        """
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= self.zfar)] = 0
        return depth

    def get_mask(self, rgb: np.ndarray):
        """Get a mask for the object in the image."""
        # HACK
        if self.mask_0:
            return self.mask_0
        else:
            raise ValueError("No mask available to return.")

    def detect(self, image_data, depth_data) -> Optional[np.ndarray]:
        logging.info(f"Processing image: {self.image_counter}")
        if not self.has_mask:
            # HACK
            # Get a mask for the object
            # Need to update this to use the mask model
            mask = self.get_mask(image_data)
            if not mask:
                logging.error("Failed to get mask.")
                return
            self.has_mask = True
            pose = self.est.register(
                K=self.K,
                rgb=image_data,
                depth=depth_data,
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )
        else:
            pose = self.est.track_one(
                rgb=image_data,
                depth=depth_data,
                K=self.K,
                iteration=self.track_refine_iter,
            )
        return pose


# TODO: How to handle different objects easily?  CLI args?
# TODO: Fix this path.
# PANCAKE_BOX_DETECTOR = FoundationPoseStream(
#     f"{CODE_DIR}/../data/rubiks_cube/mesh/cube.obj",
#     f"{CODE_DIR}/../data/rubiks_cube/cam_K.txt",
# )


@app.pose("/pancake_box/inference")
def pose_inference(data: ImageData):
    try:
        # Decode the base64 string to bytes
        image_data = base64.b64decode(data.image_base64)
        depth_data = base64.b64decode(data.depth_base64)
        # process the image
        # PANCAKE_BOX_DETECTOR.detect(image_data, depth_data)

        return {"message": "Image processed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process capture: {str(e)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
    )
    parser.add_argument(
        "--test_scene_dir",
        type=str,
        default=f"{code_dir}/demo_data/mustard0",
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--cam_k_file", type=str, default="")
    args = parser.parse_args()

    if args.cam_k_file == "":
        args.cam_k_file = f"{args.test_scene_dir}/cam_K.txt"
    mask_0 = glob.glob(f"{args.test_scene_dir}/masks/*.png")[0]
    foundation_pose_stream = FoundationPoseStream(args.mesh_file, args.cam_k_file, mask_0)
    logging.info("estimator initialization done")

    reader = datareader.YcbineoatReader(
        video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf
    )
    debug_dir = f"{CODE_DIR}/debug"
    debug = 1

    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_raw_color(i)
        depth = reader.get_raw_depth(i)
        pose = foundation_pose_stream.detect(color, depth)

        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(
            f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt",
            pose.reshape(4, 4),
        )

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(foundation_pose_stream.to_origin)
            vis = Utils.draw_posed_3d_box(
                reader.K,
                img=color,
                ob_in_cam=center_pose,
                bbox=foundation_pose_stream.bbox,
            )
            vis = Utils.draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.imshow("1", vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(
                f"{debug_dir}/track_vis/{reader.id_strs[i]}.png",
                vis,
            )


if __name__ == "__main__":
    main()
