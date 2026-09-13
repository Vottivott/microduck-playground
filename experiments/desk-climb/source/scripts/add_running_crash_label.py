"""Burn the Pollen Microduck display typography into a silent MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_label(
    frame,
    font_path: Path,
    x: int,
    y: int,
    primary_size: int,
    secondary_size: int,
):
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    primary_font = ImageFont.truetype(str(font_path), primary_size)
    secondary_font = ImageFont.truetype(str(font_path), secondary_size)

    primary = "1.6 m/s"
    secondary = "WALKING BASELINE 0.4 m/s"
    primary_bbox = draw.textbbox((x, y), primary, font=primary_font, anchor="lt")
    secondary_y = primary_bbox[3] + 8

    # A restrained two-pixel shadow keeps the requested white/gray lettering
    # legible over both the checkerboard and blue mat without adding a panel.
    shadow = (0, 0, 0, 105)
    draw.text((x + 2, y + 2), primary, font=primary_font, fill=shadow, anchor="lt")
    draw.text(
        (x + 1, secondary_y + 1),
        secondary,
        font=secondary_font,
        fill=(0, 0, 0, 95),
        anchor="lt",
    )
    draw.text(
        (x, y), primary, font=primary_font, fill=(255, 255, 255, 255), anchor="lt"
    )
    draw.text(
        (x, secondary_y),
        secondary,
        font=secondary_font,
        fill=(204, 208, 214, 255),
        anchor="lt",
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--x", type=int, default=30)
    parser.add_argument("--y", type=int, default=26)
    parser.add_argument("--primary-size", type=int, default=80)
    parser.add_argument("--secondary-size", type=int, default=23)
    args = parser.parse_args()

    reader = imageio.get_reader(args.input)
    metadata = reader.get_meta_data()
    fps = float(metadata["fps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_params=["-profile:v", "high", "-movflags", "+faststart", "-an"],
    )
    frames = 0
    try:
        for frame in reader:
            writer.append_data(
                np.asarray(draw_label(
                    frame,
                    args.font,
                    args.x,
                    args.y,
                    args.primary_size,
                    args.secondary_size,
                ))
            )
            frames += 1
    finally:
        writer.close()
        reader.close()

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "font": str(args.font),
                "font_family": "Anton",
                "font_weight": 400,
                "frames": frames,
                "fps": fps,
                "primary": "1.6 m/s",
                "secondary": "WALKING BASELINE 0.4 m/s",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
