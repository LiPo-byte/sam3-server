"""Call the watermark API (stdlib only).

Usage:
    python request.py image.png [--url http://localhost:32102]
"""
import argparse
import base64
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--url", default="http://localhost:32102")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        payload = json.dumps({"image": base64.b64encode(f.read()).decode()}).encode()

    req = urllib.request.Request(f"{args.url.rstrip('/')}/detect", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    label = "watermarked" if data["watermarked"] else "clean"
    print(f"{data['width']}x{data['height']}  {label}  score={data['score']:.3f}  {data['elapsed_ms']}ms")


if __name__ == "__main__":
    main()
