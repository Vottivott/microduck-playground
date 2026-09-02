#!/usr/bin/env python3
"""Generate the default seat plus the optional tapered battery locators."""

from pathlib import Path

import numpy as np

from generate_seat import PP, build, locator_pad_meshes


def main() -> None:
    params = dict(PP)
    params["locator_pads"] = True
    seat = build(params)
    out = Path(__file__).resolve().parent / "output_padded"
    out.mkdir(parents=True, exist_ok=True)
    seat.export(out / "param_seat_padded_mm.stl")
    meters = seat.copy()
    meters.apply_scale(0.001)
    meters.export(out / "param_seat_padded.stl")
    for index, pad_mm in enumerate(locator_pad_meshes(params)):
        pad_m = pad_mm.copy()
        pad_m.apply_scale(0.001)
        pad_m.export(out / f"locator_pad_{index}.stl")
    print(
        f"faces={len(seat.faces)} watertight={seat.is_watertight} "
        f"volume={seat.volume / 1000:.1f} cm^3"
    )
    print("bounds (mm):\n", np.round(seat.bounds, 1))
    print("exported to", out)


if __name__ == "__main__":
    main()
