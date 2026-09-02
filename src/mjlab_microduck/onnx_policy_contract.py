"""Utilities that keep exported ONNX behavior identical to training."""

from __future__ import annotations

from pathlib import Path

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
