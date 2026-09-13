# Action and handoff contract

- 50 Hz; 61 raw actor observations: base proprioception48 + commands13. Commands48:61 are zero.
- Joint order and HOME are embedded in the ONNX metadata. Resolve names; do not assume passive-joint layouts.
- Both normalized ONNX outputs are action offsets in radians. Target = HOME + executed offset; do not add another observation normalizer.
- Climber: ordinary raw outputs, no post-policy filter, base BAM gain.
- Get-up: gain multiplier0.8 relative to climbing; new executed offset = alpha×raw + (1−alpha)×previous executed offset. Alpha0.5 for neck/head,0.7 for legs.
- On switching, seed the filter with the last executed climbing offset. Keep BAM delays continuous.
- Observation slots34:48 always contain the **previous raw policy output**, including while the executed action is filtered.
- Reset policy state between episodes; do not reset on landing or when a failure flag fires during recovery.

`runtime/policy_pair.py` provides a CPU ONNX reference for this action contract. It requires a caller-supplied `eligible` flag and returns the target and requested gain ratio. It does **not** implement robot arming, motor control, or a deployable desk detector.

The recorded supervisor checks direct foot–desktop contact and ladder-relative root position at least4cm inside the desktop. Its geometry information is privileged simulator state. A future onboard detector must be validated for early/missed switches before treating this as a hardware skill.

GPU rollouts can diverge even from matching initial states. The included CPU action replay tests inference/conversion against recorded observations, not equality of independent physics trajectories.

Validation: CPU checkpoint versus ONNX maximum errors were2.63e-6 (climber) and9.54e-7 (get-up) over300 recorded observations each. Across all3,000 selected-trial steps, CPU ONNX versus the recorded GPU actions differed by up to0.00258rad with recorded history, and0.00335rad when feeding back the CPU outputs. This is close numerical agreement, not bit-exact recorded action parity.
