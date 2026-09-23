"""Burn the live climb height into a corridor-climb clip.

Top left, big white type with a dark drop shadow - the same treatment as the
swing-angle and winner overlays.  Below a metre it reads in whole centimetres
("27cm"); at or above a metre it reads in metres to one decimal ("1.2m").

The height is the trunk's height above the floor, recorded per frame by
``render_checkpoint.py --height-track robot`` into ``heights.json`` beside the
clip.  A corridor has nothing in it to give a video scale, so without this the
only cue is the wall marks scrolling past.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def label_for(height_m: float) -> str:
    cm = int(round(height_m * 100))
    # switch on the ROUNDED value, or 0.999 m reads "100cm"
    if cm < 100:
        return f"{cm}cm"
    return f"{height_m:.1f}m"


def _font(path: str | None, size: int) -> ImageFont.ImageFont:
    if path:
        return ImageFont.truetype(path, size)
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    try:  # matplotlib ships DejaVu and is always installed here
        import matplotlib

        return ImageFont.truetype(
            str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"), size
        )
    except Exception:
        return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--heights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--font", type=str, default="")
    ap.add_argument("--font-size", type=int, default=64)
    ap.add_argument("--x", type=int, default=28)
    ap.add_argument("--y", type=int, default=20)
    args = ap.parse_args()

    data = json.loads(args.heights.read_text())
    heights = data["heights_m"]
    reader = imageio.get_reader(args.input)
    fps = float(reader.get_meta_data()["fps"])
    font = _font(args.font or None, args.font_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=fps, codec="libx264", quality=9, pixelformat="yuv420p",
        ffmpeg_params=["-profile:v", "high", "-movflags", "+faststart", "-an"],
    )
    try:
        for i, frame in enumerate(reader):
            # the recorder may drop or repeat a frame; clamp rather than fail
            h = heights[min(i, len(heights) - 1)]
            image = Image.fromarray(np.asarray(frame)).convert("RGB")
            draw = ImageDraw.Draw(image)
            text = label_for(float(h))
            draw.text((args.x + 3, args.y + 3), text, font=font, fill=(0, 0, 0))
            draw.text((args.x, args.y), text, font=font, fill=(255, 255, 255))
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
        reader.close()
    print(f"OVERLAY_DONE {args.output} frames={i + 1}")


if __name__ == "__main__":
    main()
