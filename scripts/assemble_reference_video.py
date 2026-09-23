"""Join contiguous renders of one physics recording, with the usual height text."""
import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

from add_climb_height_overlay import _font, label_for

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("parts", nargs="+", type=Path)
p.add_argument("--fps", type=int, default=25)
a = p.parse_args()
font = _font(None, 64)
audits = []
frame_count = 0
with iio.get_writer(a.output, fps=a.fps, codec="libx264", quality=8,
                    macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"]) as writer:
    for part in a.parts:
        heights = json.loads(part.with_name(part.stem + "_heights.json").read_text())["heights_m"]
        audit = json.loads(part.with_name(part.stem + "_audit.json").read_text())
        assert len(audit) == len(heights) == 150
        audits.extend(audit)
        with iio.get_reader(part) as reader:
            count = 0
            for frame, height in zip(reader, heights, strict=True):
                assert frame.shape == (720, 720, 3)
                im = Image.fromarray(frame)
                draw = ImageDraw.Draw(im)
                label = label_for(height)
                draw.text((31, 23), label, font=font, fill="black")
                draw.text((28, 20), label, font=font, fill="white")
                writer.append_data(np.asarray(im))
                frame_count += 1
                count += 1
            assert count == 150
assert frame_count == 36 * a.fps
times = np.array([v["time"] for v in audits])
assert np.allclose(times, np.arange(frame_count) / a.fps)
ending = [v for v in audits if v["time"] >= 33]
assert all(all(v["feet_on_platform"]) for v in ending)
assert max(v["tilt_deg"] for v in ending) < 2
assert min(v["carriage_m"] for v in ending) > .18
assert max(v["speed_mps"] for v in ending) < .02
a.output.with_suffix(".audit.json").write_text(json.dumps(audits, indent=2))
print("COMPLETE_REFERENCE_VIDEO", a.output, frame_count, "frames; final standing audit passed")
