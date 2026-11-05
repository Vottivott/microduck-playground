"""MicroDock v2 (the seated dock) invariants: it is the MicroDock with three documented changes, inside the ranges the
seated docking and undocking results depend on."""

import dataclasses
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hardware/microdock/source"))
_SPEC = importlib.util.spec_from_file_location("microdock_v2_seated", ROOT / "hardware/microdock-v2/source/seated.py")
seated = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(seated)
P, P1 = seated.Params(), seated.dk.Params()
CHANGED = {"z0", "hood_float_y", "hood_drop"}


def test_only_the_documented_parameters_change():
    differ = {f.name for f in dataclasses.fields(P1) if getattr(P, f.name) != getattr(P1, f.name)}
    assert differ == CHANGED


def test_the_hood_hangs_at_the_seated_height():
    assert 50.0 <= P.z0 <= 62.0 < P1.z0          # seated trunk height when docked (the standing dock: 118.6)


def test_the_hood_floats_further_than_on_the_microdock():
    assert P.hood_float_y > P1.hood_float_y and P.hood_drop > P1.hood_drop


def test_the_adapter_sits_at_the_ports_height():
    assert P.adapter_dz == 0.0                   # the docking policy sits level (an older, leaning one needed -2.3 mm)
    assert P.adapter_z == P.port_z and P1.adapter_z == P1.port_z


def test_the_policies_ship_with_the_dock():
    for name in ("scoot_dock.onnx", "scoot_undock.onnx", "experimental/scoot_dock_undock.onnx"):
        policy = ROOT / "hardware/microdock-v2/policy" / name
        assert policy.exists() and policy.stat().st_size > 100_000, name
