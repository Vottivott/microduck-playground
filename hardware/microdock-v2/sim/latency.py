"""The training's latencies for the docking simulation.

The butt walks were trained with mjlab's actuator delay (servo targets 3 to 6 physics steps late, resampled every
physics step), the joint velocities one control step late and the IMU (gyro, gravity) zero to one step late. The
simulation's control loop (docking.step_control) has none, and the undocking policy only breaks free of the magnet with
them: it pushes in the rhythm the delays set. enable() swaps in a delayed control loop and delayed observations (the
IMU always one step late):
- official: the official policies' phases (walk, turn, sit, stand);
- butt: the butt walks' phases (backing in, scooting out).
Call it after docking_seated.make(). The real robot's latency has not been measured.
"""
import numpy as np, mujoco
import docking_seated as ds
dg, dk = ds.dg, ds.dk


def enable(pol, bc, seed=0, official=True, butt=True):
    """official / butt: delay the official policies' phases / the butt walks'. Returns a function that clears the delay
    lines (docking.reset calls it too)."""
    rng = np.random.default_rng(seed); tq = [bc.q_target.copy()]*8; hist = []
    orig = pol.get_observations
    def on():
        return butt if pol.ort_session is getattr(pol, "butt_walk_session", None) or getattr(pol, "undock_session", None) is pol.ort_session else official
    def delayed():                                     # the IMU (0:6) and the joint velocities (20:34) one control step late
        o = orig(); hist.append(o.copy()); del hist[:-2]; o = o.copy()
        if on(): o[0:6] = hist[0][0:6]; o[20:34] = hist[0][20:34]
        return o
    pol.get_observations = delayed
    def step_control(model, data, bam_ctrl, pol_, frame_cb=None):
        pol_.update_behavior(0.02)
        a = pol_.infer(); tgt = pol_.default_pose + a*pol_.action_scale
        for k in range(4):                             # four physics steps per control step; targets 3-6 steps late
            tq.append(tgt.copy()); del tq[:-8]
            bam_ctrl.q_target[:] = tq[max(0, len(tq) - 1 - int(rng.integers(3, 7)))] if on() else tgt
            bam_ctrl.update(); mujoco.mj_step(model, data)
            if frame_cb: frame_cb(k)
    def reset_hook():
        tq[:] = [bc.q_target.copy()]*8; hist.clear()
    dg.step_control = step_control
    _reset = dg.reset
    def reset(*a, **k):
        _reset(*a, **k); reset_hook()
    dg.reset = reset
    return reset_hook
