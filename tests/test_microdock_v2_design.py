"""MicroDock v2 (the seated dock) invariants: it is the MicroDock with four documented changes, inside the ranges the
30/30 seated docking result depends on."""

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
CHANGED = {"z0", "hood_float_y", "hood_drop", "adapter_dz"}


def test_only_the_documented_parameters_change():
    differ = {f.name for f in dataclasses.fields(P1) if getattr(P, f.name) != getattr(P1, f.name)}
    assert differ == CHANGED


def test_the_hood_hangs_at_the_seated_height():
    assert 50.0 <= P.z0 <= 62.0 < P1.z0          # seated trunk height when docked (the standing dock: 118.6)


def test_the_hood_floats_further_than_on_the_microdock():
    assert P.hood_float_y > P1.hood_float_y and P.hood_drop > P1.hood_drop


def test_the_adapter_offset_stays_inside_the_magnets_reach():
    assert -2.5 < P.adapter_dz < 0.0             # dock.connected() accepts |vertical| < 2.5 mm
    assert P.adapter_z == P.port_z + P.adapter_dz and P1.adapter_z == P1.port_z


def test_the_butt_walk_policy_ships_with_the_dock():
    policy = ROOT / "hardware/microdock-v2/policy/butt_walk.onnx"
    assert policy.exists() and policy.stat().st_size > 100_000
