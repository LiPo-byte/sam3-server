"""Call the SAM3 API and write an SVG overlay for visual check (stdlib only).

Usage:
    python request.py image.png sofa lamp [--url http://<host>:32100] [--out result.svg]
"""
import argparse
import base64
import json
import mimetypes
import urllib.request

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--url", default="http://localhost:32100")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="result.svg")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    payload = json.dumps({
        "image": base64.b64encode(image_bytes).decode(),
        "prompts": args.prompts,
        "threshold": args.threshold,
    }).encode()
    req = urllib.request.Request(f"{args.url}/segment", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    print(f"{data['width']}x{data['height']}  {data['elapsed_ms']}ms  {len(data['results'])} results")
    for r in data["results"]:
        print(f"  {r['prompt']:<15} score={r['score']:.3f} box={r['box']} path_len={len(r['path'])}")

    mime = mimetypes.guess_type(args.image)[0] or "image/png"
    href = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    color_of = {p: COLORS[i % len(COLORS)] for i, p in enumerate(args.prompts)}
    paths = "\n".join(
        f'  <path d="{r["path"]}" fill="{color_of.get(r["prompt"], "#ff0")}" fill-opacity="0.45" '
        f'fill-rule="evenodd" stroke="{color_of.get(r["prompt"], "#ff0")}" stroke-width="2"/>'
        for r in data["results"]
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{data["width"]}" height="{data["height"]}" '
        f'viewBox="0 0 {data["width"]} {data["height"]}">\n'
        f'  <image href="{href}" width="{data["width"]}" height="{data["height"]}"/>\n{paths}\n</svg>\n'
    )
    with open(args.out, "w") as f:
        f.write(svg)
    print(f"overlay written to {args.out}")


if __name__ == "__main__":
    main()
