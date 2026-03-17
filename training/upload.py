"""
This module provides functionality to upload model files to Pinata IPFS using their API.
"""

import os
import json
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

PINATA_JWT = os.getenv("PINATA_JWT")
UPLOAD_URL = "https://uploads.pinata.cloud/v3/files"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_model(filepath: str, name: str = None, keyvalues: dict = None) -> str:
    """Upload a model file to Pinata IPFS. Returns the CID."""
    if not PINATA_JWT:
        raise EnvironmentError("Missing PINATA_JWT environment variable")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f)}
        data = {"network": "public"}
        if name:
            data["name"] = name
        if keyvalues:
            data["keyvalues"] = json.dumps(keyvalues)

        response = requests.post(
            UPLOAD_URL,
            headers={"Authorization": f"Bearer {PINATA_JWT}"},
            files=files,
            data=data,
            timeout=60,
        )

    if response.status_code != 200:
        raise requests.HTTPError(f"Upload failed: {response.status_code} - {response.text}")

    result = response.json()
    if "data" not in result or "cid" not in result["data"]:
        raise ValueError(f"Unexpected response format: {result}")

    return result["data"]["cid"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a model file to Pinata IPFS")
    parser.add_argument("--model", required=True, help="Path to the model file (e.g. model.onnx)")
    parser.add_argument("--name", help="Optional display name on Pinata")
    args = parser.parse_args()

    cid = upload_model(args.model, name=args.name)
    print(f"CID: {cid}")
