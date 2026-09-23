"""Utilities that keep exported ONNX behavior identical to training."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import onnx
from onnx import helper, numpy_helper


def bake_action_clip(onnx_path: str | Path, clip_actions: float | None) -> None:
    """Append the RSL-RL action clamp to an exported policy graph.

    ``RslRlVecEnvWrapper`` clips actor outputs before applying actions during
    training. The deployment runtime consumes ONNX outputs directly, so the
    clamp belongs in the exported graph whenever ``clip_actions`` is finite.
    """
    if clip_actions is None:
        return
    clip_actions = float(clip_actions)
    if not np.isfinite(clip_actions):
        return
    if clip_actions <= 0.0:
        raise ValueError("clip_actions must be positive")

    path = Path(onnx_path)
    model = onnx.load(path)
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError(f"Expected one policy output, found {len(graph.output)}")
    if any(node.name == "action_clip" for node in graph.node):
        raise ValueError("ONNX policy already contains the action_clip node")

    output_name = graph.output[0].name
    unclipped_name = f"{output_name}_unclipped"
    producers = []
    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == output_name:
                producers.append((node, index))
    if len(producers) != 1:
        raise ValueError(
            f"Expected one producer for output {output_name!r}, found {len(producers)}"
        )
    producer, output_index = producers[0]
    producer.output[output_index] = unclipped_name

    min_name = "action_clip_min"
    max_name = "action_clip_max"
    existing_names = {
        value.name for value in graph.initializer
    } | {name for node in graph.node for name in (*node.input, *node.output)}
    if min_name in existing_names or max_name in existing_names:
        raise ValueError("ONNX policy already uses reserved action-clip names")
    graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray(-clip_actions, dtype=np.float32), name=min_name
            ),
            numpy_helper.from_array(
                np.asarray(clip_actions, dtype=np.float32), name=max_name
            ),
        ]
    )
    graph.node.append(
        helper.make_node(
            "Clip",
            [unclipped_name, min_name, max_name],
            [output_name],
            name="action_clip",
        )
    )

    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update(
        {
            "action_clip": f"[-{clip_actions:g}, {clip_actions:g}]",
            "action_clip_baked_into_onnx": "true",
        }
    )
    del model.metadata_props[:]
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, path)

def bake_action_bounds(onnx_path: str | Path, lo, hi, metadata_note: str = "") -> None:
    """Append a per-output clamp ``min(max(a, lo), hi)`` to the policy graph.

    Used for policies trained with ``BoundedJointPositionAction``: the training
    env clamps the commanded joint target to the joint travel and feeds the
    bounded command back into the action history, so the deployed graph must
    apply the identical clamp (in policy-output units) for the runtime's
    action history to match training.
    """
    lo = np.asarray(lo, dtype=np.float32).reshape(-1)
    hi = np.asarray(hi, dtype=np.float32).reshape(-1)
    if lo.shape != hi.shape or lo.size == 0:
        raise ValueError("lo/hi must be non-empty and equally shaped")
    if np.any(lo > hi):
        raise ValueError("lo must be <= hi for every action")

    path = Path(onnx_path)
    model = onnx.load(path)
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError(f"Expected one policy output, found {len(graph.output)}")
    if any(node.name in ("action_bounds_max", "action_bounds_min") for node in graph.node):
        raise ValueError("ONNX policy already contains the action bound nodes")

    output_name = graph.output[0].name
    unbounded_name = f"{output_name}_unbounded"
    producers = [(node, i) for node in graph.node for i, name in enumerate(node.output) if name == output_name]
    if len(producers) != 1:
        raise ValueError(f"Expected one producer for output {output_name!r}, found {len(producers)}")
    producer, output_index = producers[0]
    producer.output[output_index] = unbounded_name

    lo_name, hi_name, mid_name = "action_bounds_lo", "action_bounds_hi", f"{output_name}_lower_bounded"
    graph.initializer.extend([
        onnx.numpy_helper.from_array(lo, name=lo_name),
        onnx.numpy_helper.from_array(hi, name=hi_name),
    ])
    graph.node.extend([
        onnx.helper.make_node("Max", [unbounded_name, lo_name], [mid_name], name="action_bounds_max"),
        onnx.helper.make_node("Min", [mid_name, hi_name], [output_name], name="action_bounds_min"),
    ])
    for key, value in {
        "action_bounds_lo": json.dumps([round(float(v), 6) for v in lo]),
        "action_bounds_hi": json.dumps([round(float(v), 6) for v in hi]),
        "action_bounds_baked_into_onnx": "true",
        "action_bounds_note": metadata_note or "joint-position commands clamped to the joint travel (policy-output units)",
    }.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, path)
