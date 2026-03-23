"""
Pinata IPFS client — upload, download, delete, and update model files.

All operations use the Pinata v3 Files API.
Requires env vars (via .env or shell):
  PINATA_JWT      — Bearer token (required for all operations)
  PINATA_GATEWAY  — Your dedicated gateway hostname, e.g. "myapp.mypinata.cloud"
                    Falls back to the public gateway if not set.
"""

import os
import json
import requests
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

PINATA_JWT = os.getenv("PINATA_JWT")
PINATA_GATEWAY = os.getenv("PINATA_GATEWAY", "gateway.pinata.cloud")

_FILES_API = "https://api.pinata.cloud/v3/files"
_UPLOAD_URL = "https://uploads.pinata.cloud/v3/files"


def _auth_headers() -> dict:
    if not PINATA_JWT:
        raise EnvironmentError("Missing PINATA_JWT environment variable")
    return {"Authorization": f"Bearer {PINATA_JWT}"}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload(filepath: str, name: str = None, keyvalues: dict = None) -> dict:
    """
    Upload a file to Pinata IPFS.

    Returns a dict with at least:
      { "id": ..., "cid": ..., "name": ..., "size": ..., "created_at": ... }
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        form_data = {"network": "public"}
        if name:
            form_data["name"] = name
        if keyvalues:
            form_data["keyvalues"] = json.dumps(keyvalues)

        resp = requests.post(
            _UPLOAD_URL,
            headers=_auth_headers(),
            files={"file": (os.path.basename(filepath), f)},
            data=form_data,
            timeout=120,
        )

    _raise_for_status(resp, "Upload failed")
    data = resp.json()["data"]
    return data


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download(cid: str, dest: str) -> str:
    """
    Download a file by CID from the Pinata gateway.

    Saves to `dest` (path to file, not directory).
    Returns the absolute path of the saved file.
    """
    url = f"https://{PINATA_GATEWAY}/ipfs/{cid}"
    resp = requests.get(url, headers=_auth_headers(), stream=True, timeout=120)
    _raise_for_status(resp, f"Download failed for CID {cid}")

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(dest_path.resolve())


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_metadata(file_id: str) -> dict:
    """
    Fetch full metadata for a file by its Pinata file ID (not CID).

    Returns the file object dict (id, cid, name, size, keyvalues, created_at, ...).
    """
    resp = requests.get(
        f"{_FILES_API}/{file_id}",
        headers=_auth_headers(),
        timeout=30,
    )
    _raise_for_status(resp, f"get_metadata failed for id {file_id}")
    return resp.json()["data"]


def update_metadata(file_id: str, name: str = None, keyvalues: dict = None) -> dict:
    """
    Update the name and/or keyvalues of a file by its Pinata file ID.

    Returns the updated file object.
    """
    body = {}
    if name is not None:
        body["name"] = name
    if keyvalues is not None:
        body["keyvalues"] = keyvalues
    if not body:
        raise ValueError("Provide at least one of: name, keyvalues")

    resp = requests.put(
        f"{_FILES_API}/{file_id}",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=30,
    )
    _raise_for_status(resp, f"update_metadata failed for id {file_id}")
    return resp.json()["data"]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete(file_id: str) -> None:
    """
    Unpin and delete a file from Pinata by its file ID.

    Note: the file's CID may still be accessible via other pinning services
    until garbage-collected globally. On Pinata, deletion removes your pin.
    """
    resp = requests.delete(
        f"{_FILES_API}/{file_id}",
        headers=_auth_headers(),
        timeout=30,
    )
    _raise_for_status(resp, f"delete failed for id {file_id}")


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_files(name_contains: str = None, keyvalues: dict = None, limit: int = 10) -> list:
    """
    List files on Pinata, with optional filtering.

    Returns a list of file objects.
    """
    params = {"limit": limit}
    if name_contains:
        params["name"] = name_contains
    if keyvalues:
        for k, v in keyvalues.items():
            params[f"keyvalues[{k}]"] = v

    resp = requests.get(
        _FILES_API,
        headers=_auth_headers(),
        params=params,
        timeout=30,
    )
    _raise_for_status(resp, "list_files failed")
    return resp.json()["data"]["files"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raise_for_status(resp: requests.Response, msg: str) -> None:
    if not resp.ok:
        raise requests.HTTPError(f"{msg}: HTTP {resp.status_code} — {resp.text}")


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Pinata IPFS client")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload", help="Upload a file")
    p_up.add_argument("file")
    p_up.add_argument("--name")
    p_up.add_argument("--kv", nargs="*", metavar="KEY=VALUE", help="keyvalues, e.g. arch=mobilenet version=1")

    p_dl = sub.add_parser("download", help="Download a file by CID")
    p_dl.add_argument("cid")
    p_dl.add_argument("dest")

    p_meta = sub.add_parser("metadata", help="Get metadata by file ID")
    p_meta.add_argument("file_id")

    p_upd = sub.add_parser("update", help="Update metadata by file ID")
    p_upd.add_argument("file_id")
    p_upd.add_argument("--name")
    p_upd.add_argument("--kv", nargs="*", metavar="KEY=VALUE")

    p_del = sub.add_parser("delete", help="Delete a file by file ID")
    p_del.add_argument("file_id")

    p_ls = sub.add_parser("list", help="List files")
    p_ls.add_argument("--name")
    p_ls.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    def parse_kv(pairs):
        if not pairs:
            return None
        result = {}
        for pair in pairs:
            k, _, v = pair.partition("=")
            result[k.strip()] = v.strip()
        return result

    if args.cmd == "upload":
        result = upload(args.file, name=args.name, keyvalues=parse_kv(args.kv))
        print(f"Uploaded successfully:")
        pprint.pprint(result)
        print(f"\nCID:     {result['cid']}")
        print(f"File ID: {result['id']}")

    elif args.cmd == "download":
        path = download(args.cid, args.dest)
        print(f"Saved to: {path}")

    elif args.cmd == "metadata":
        pprint.pprint(get_metadata(args.file_id))

    elif args.cmd == "update":
        result = update_metadata(args.file_id, name=args.name, keyvalues=parse_kv(args.kv))
        print("Updated:")
        pprint.pprint(result)

    elif args.cmd == "delete":
        delete(args.file_id)
        print(f"Deleted file {args.file_id}")

    elif args.cmd == "list":
        files = list_files(name_contains=args.name, limit=args.limit)
        print(f"{len(files)} file(s):")
        for f in files:
            print(f"  {f['id']}  cid={f['cid']}  name={f.get('name', '')}  size={f.get('size', '?')}B")
