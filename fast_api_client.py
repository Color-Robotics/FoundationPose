import argparse
import glob
import os
import base64
import io
import requests

from PIL import Image

def encode_image_to_base64(image_fn):
    # Encode the image to base64
    image = Image.open(image_fn)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def send_image_to_endpoint(rgb_base64, depth_base64):
    # Assume a tunnel is built to the FastAPI server.
    url = 'http://127.0.0.1:8000/rubikscube/inference'
    headers = {'Content-Type': 'application/json'}
    data = {"image_base64": rgb_base64, "depth_base64": depth_base64}
    response = requests.post(url, json=data, headers=headers)
    return response

def main():
    # Argparse with argument data dir
    parser = argparse.ArgumentParser(description='FastAPI client')
    parser.add_argument('data_dir', type=str, help='Data directory')
    args = parser.parse_args()

    # Send a POST request to the /api/run-pose-estimation endpoint
    for rgb_image in glob.glob(os.path.join(args.data_dir, 'rgb', '*.png')):
        # V hacky
        depth_image = rgb_image.replace("rgb", "depth")
        import pdb; pdb.set_trace()
        image_base64 = encode_image_to_base64(rgb_image)
        depth_base64 = encode_image_to_base64(depth_image)
        response = send_image_to_endpoint(image_base64, depth_base64)
        # Should get back a pose (3x3 matrix)
        print(f"For image {rgb_image}, pose: {response.json()}")

if __name__ == "__main__":
    main()
