"""Call the PP-OCR API and write an SVG overlay for visual check (stdlib only).

Usage:
    python request.py image.png [--url http://localhost:32101] [--out result.svg]
"""
import argparse
import base64
import html
import json
import mimetypes
import urllib.request


def _points(poly, box):
    if poly:
        return " ".join(f"{p[0]},{p[1]}" for p in poly)
    if len(box) == 4:
        x1, y1, x2, y2 = box
        return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--url", default="http://localhost:32101")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--out", default="result.svg")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    payload = json.dumps({
        "image": base64.b64encode(image_bytes).decode(),
        "score_threshold": args.score_threshold,
    }).encode()
    req = urllib.request.Request(f"{args.url}/ocr", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    print(f"{data['width']}x{data['height']}  {data['elapsed_ms']}ms  {len(data['results'])} lines")
    shapes = []
    for r in data["results"]:
        print(f"  {r['score']:.3f}  {r['text']}")
        points = _points(r.get("poly") or [], r.get("box") or [])
        if not points:
            continue
        label = html.escape(r["text"])
        x = r["box"][0] if r.get("box") else r["poly"][0][0]
        y = r["box"][1] if r.get("box") else r["poly"][0][1]
        shapes.append(
            f'  <polygon points="{points}" fill="#e6194b" fill-opacity="0.25" '
            f'stroke="#e6194b" stroke-width="2"/>\n'
            f'  <text x="{x}" y="{y}" fill="#e6194b" font-size="16">{label}</text>'
        )

    mime = mimetypes.guess_type(args.image)[0] or "image/png"
    href = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{data["width"]}" height="{data["height"]}" '
        f'viewBox="0 0 {data["width"]} {data["height"]}">\n'
        f'  <image href="{href}" width="{data["width"]}" height="{data["height"]}"/>\n'
        + "\n".join(shapes) + "\n</svg>\n"
    )
    with open(args.out, "w") as f:
        f.write(svg)
    print(f"overlay written to {args.out}")


if __name__ == "__main__":
    main()
