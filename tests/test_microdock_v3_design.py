"""MicroDock v3 (the egg dock) invariants: the simulated dock is the printed one (the stage's springs, the magnetic pair,
the stack's front stop, where the cable leaves), and the published results and files are the ones the README cites."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "hardware/microdock-v3"
sys.path.insert(0, str(V3 / "source"))
import flexdock as fd  # noqa: E402
import flexdock_c as fc  # noqa: E402

RESULTS = json.loads((V3 / "results/egg_dock_trials.json").read_text())
# the print's parameters (flexdock_c_build.V4 with the magnetic pair as it is)
BUILD = dict(k_y=67.0, k_z=97.0, z_neutral=5.0, z_travel=15.0, hood_mass=0.083)
P = fd.FlexDock(**BUILD, **fc.PLUG_C)


def flags(command):
    return dict(re.findall(r"--(\w+)[ =](\S+)", command))


def test_the_build_prints_the_simulated_springs():
    build = (V3 / "source/flexdock_c_build.py").read_text()
    assert "V4 = dict(k_y=67.0, k_z=97.0, z_neutral=5.0, z_travel=15.0, x_solref=0.02, hood_mass=0.083)" in build
    for batch in ("normal_cable", "stiff_cable"):
        f = flags(RESULTS[batch]["command"])
        assert f["design"] == "flex"
        for k, v in BUILD.items():
            assert float(f[k]) == v, (batch, k)


def test_the_simulated_magnetic_pair_and_stack_are_the_printed_ones():
    for batch in ("normal_cable", "stiff_cable"):
        f = flags(RESULTS[batch]["command"])
        out, fw, fh, fl, bw, bh = (float(v) for v in f["tip"].split(","))
        assert (out, bw, bh, (fw, fh, fl)) == (fc.PLUG_C["tip_out"], fc.PLUG_C["tip_w"], fc.PLUG_C["tip_h"], fc.PLUG_C["tip_front"])
        assert tuple(float(v) for v in f["plug"].split(",")) == fc.PLUG_C["plug"]
        assert float(f["face_offset"]) == fc.FLOAT["fwd"]          # the adapter at the stack's front stop
        cable_out = [float(v) for v in f["cable_out"].split(",")]
        assert all(abs(a - b) <= 0.1 for a, b in zip(cable_out, fc.goobay_cable_out(P)))


def test_the_stack_floats_between_its_stops():
    x = fc.float_x(P)
    assert abs(x["face_fwd"] - x["face_back"] - 1.45) < 1e-9
    assert x["rise_gap"] > 0.25                                    # it goes in from below, held forward


def test_the_published_results():
    for batch, stood in (("normal_cable", 90), ("stiff_cable", 89)):
        total = RESULTS[batch]["total"]
        assert (total["runs"], total["docked"], total["usb_drops"], total["released"]) == (90, 90, 0, 90), batch
        assert (total["stood"], total["walked"]) == (stood, 90), batch
        assert len(RESULTS[batch]["trials"]) == 90


def test_the_print_set_and_the_manual_ship():
    names = ("base", "front", "dome", "stage", "hood", "tag", "test")
    for k, name in enumerate(names, start=1):
        assert (V3 / f"stl/microdock_v3_{name}.stl").exists(), name
        for ext in ("stl", "png"):
            assert (V3 / f"print/microdock_v3_plate{k}_{name}.{ext}").exists(), (k, ext)
    assert (V3 / "print/microdock_v3_P1S.3mf").stat().st_size > 1_000_000
    assert (V3 / "media/microdock_v3_assembly.pdf").stat().st_size > 1_000_000
    assert len(list((V3 / "media/assembly").glob("page_*.jpg"))) == 8
