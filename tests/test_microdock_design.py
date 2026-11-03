"""MicroDock design invariants (hardware/microdock/source/dock.py): the clearances the 30/30 docking result depends on."""

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "microdock_dock", Path(__file__).resolve().parents[1] / "hardware/microdock/source/dock.py"
)
dock = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dock)
P = dock.Params()


def test_skids_leave_the_power_button_and_lights_free():
    collar_edge = abs(P.button_xy[1]) + P.button_r + 1.2
    led_edge = max(abs(y) for _, y in P.leds_xy) + 1.5
    assert collar_edge < P.skid_y and led_edge < P.skid_y
    for name in ("skidL", "skidR", "lipL", "lipR"):
        y = dock.hood_pieces(P)[name][:, 1]
        assert np.abs(y).min() >= P.skid_y - 1e-9, name


def test_side_walls_line_up_the_body_with_a_small_clearance():
    clearance = P.side_in - P.trunk_half_w
    assert 0.3 <= clearance <= 1.5


def test_adapter_waits_just_behind_the_magnetic_tip():
    assert P.adapter_face_x < P.tip_face_x < P.shell_back_x
    assert P.tip_face_x - P.adapter_face_x <= 0.5
    assert P.hood_rear < P.adapter_face_x - P.plug[2] - P.angle_plug[1]  # room for the cable head behind the adapter


def test_battery_stop_limits_how_far_the_tip_is_pressed():
    tip_contact = P.tip_face_x - P.adapter_face_x  # backward travel past nominal until the tip meets the adapter
    stop_contact = P.back_face_x - P.stop_face_x  # ... until the battery meets the hood's stop
    assert 0.0 < tip_contact < stop_contact <= tip_contact + 0.5


def test_connected_window_lies_inside_the_magnet_reach():
    assert dock.connected([0.0, 0.0, 0.0]) and dock.connected([1.9, 2.4, -2.4])
    assert not dock.connected([2.1, 0.0, 0.0]) and not dock.connected([0.0, 2.6, 0.0])
    assert 2.0 <= dock.MAG_RANGE and 2.5 <= dock.MAG_LAT
