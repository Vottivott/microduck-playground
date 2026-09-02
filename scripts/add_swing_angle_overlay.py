"""Crop a swing video and burn in its live running peak-to-peak angle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def running_span_degrees(samples: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([sample["time_s"] for sample in samples], dtype=np.float64)
    angles = np.asarray([sample["angle_rad"] for sample in samples], dtype=np.float64)
    spans = np.degrees(np.maximum.accumulate(angles) - np.minimum.accumulate(angles))
    return times, spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--crop-x", type=int, default=160)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=960)
    parser.add_argument("--crop-height", type=int, default=720)
    parser.add_argument("--x", type=int, default=30)
    parser.add_argument("--y", type=int, default=26)
    parser.add_argument("--font-size", type=int, default=80)
    parser.add_argument("--green-hold-s", type=float, default=0.24)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    metric_times, running_spans = running_span_degrees(metrics["samples"])

    reader = imageio.get_reader(args.input)
    metadata = reader.get_meta_data()
    fps = float(metadata["fps"])
    font = ImageFont.truetype(str(args.font), args.font_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=fps,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
        ffmpeg_params=["-profile:v", "high", "-movflags", "+faststart", "-an"],
    )

    frame_count = 0
    last_displayed_span = 0
    green_until_s = -math.inf
    final_displayed_span = 0
    try:
        for frame_count, frame in enumerate(reader, start=1):
            frame_time_s = (frame_count - 1) / fps
            metric_index = int(
                np.clip(
                    np.searchsorted(metric_times, frame_time_s, side="right") - 1,
                    0,
                    len(metric_times) - 1,
                )
            )
            displayed_span = round(float(running_spans[metric_index]))
            if displayed_span > last_displayed_span and frame_count > 1:
                green_until_s = frame_time_s + args.green_hold_s
            last_displayed_span = max(last_displayed_span, displayed_span)
            final_displayed_span = last_displayed_span

            x0, y0 = args.crop_x, args.crop_y
            x1, y1 = x0 + args.crop_width, y0 + args.crop_height
            cropped = np.asarray(frame)[y0:y1, x0:x1]
            if cropped.shape[:2] != (args.crop_height, args.crop_width):
                raise ValueError(
                    f"Crop produced {cropped.shape[:2]}, expected "
                    f"{(args.crop_height, args.crop_width)}"
                )

            image = Image.fromarray(cropped).convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            label = f"{last_displayed_span}{chr(176)}"
            color = (
                (0, 174, 66, 255)
                if frame_time_s <= green_until_s
                else (255, 255, 255, 255)
            )
            draw.text(
                (args.x + 2, args.y + 2),
                label,
                font=font,
                fill=(0, 0, 0, 110),
                anchor="lt",
            )
            draw.text(
                (args.x, args.y), label, font=font, fill=color, anchor="lt"
            )
            writer.append_data(np.asarray(Image.alpha_composite(image, overlay).convert("RGB")))
    finally:
        writer.close()
        reader.close()

    print(
        json.dumps(
            {
                "input": str(args.input),
                "metrics": str(args.metrics),
                "output": str(args.output),
                "frames": frame_count,
                "fps": fps,
                "crop": {
                    "x": args.crop_x,
                    "y": args.crop_y,
                    "width": args.crop_width,
                    "height": args.crop_height,
                },
                "font": str(args.font),
                "font_family": "Anton",
                "font_size": args.font_size,
                "green": "#00AE42",
                "green_hold_s": args.green_hold_s,
                "final_displayed_span_deg": final_displayed_span,
                "audio": "none",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
