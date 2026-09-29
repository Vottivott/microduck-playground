# Release verification — 2026-09-29

Both rendered checkpoint files were freshly downloaded from their HF repos and SHA-256 matched. Unmodified release profiles, seed 28, one environment, 50 Hz, CPU MuJoCo Warp with OSMesa; original source revision 58d56df0e863df552fac7caf409aaa968cfde34d. This split preserves those two profiles and physics modules unchanged; only unpublished pillar profiles are removed.

- Long jump: crosses the gap, lands, recovers and stays on the target through 12 seconds (timeout).
- Backflip: backward rotation, landing and standing recovery, followed by drifting off the mat and falling at 7.62 seconds. Not a stable idle policy.

These are visual single-start observations, not robustness estimates or exact replay of the original edited footage. CPU/GPU random streams and numerical differences can change trajectories. Earlier archived checks established full-state continuation and normalized ONNX parity on GPU; smoke tests do not establish maneuver success.

Fresh termination records are in `eval/render-*-20260929.json`; original diagnostic/parity records are retained separately. No policy weights were changed. Source links and publication metadata are updated for this two-policy release.
