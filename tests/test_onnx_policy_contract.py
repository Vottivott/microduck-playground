"""Regression tests for deployment transforms baked into ONNX policies."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from mjlab_microduck.onnx_policy_contract import bake_action_clip


def _write_linear_policy(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "test-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.save(model, path)


def test_action_clip_is_baked_into_policy_output(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_policy(path)
    bake_action_clip(path, 1.0)

    model = onnx.load(path)
    assert [(node.name, node.op_type) for node in model.graph.node][-1] == (
        "action_clip",
        "Clip",
    )
    metadata = {item.key: item.value for item in model.metadata_props}
    assert metadata["action_clip"] == "[-1, 1]"
    assert metadata["action_clip_baked_into_onnx"] == "true"

    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"obs": np.asarray([[4.5, -3.0]], np.float32)})[0]
    np.testing.assert_array_equal(actual, [[1.0, -1.0]])


def test_action_clip_rejects_invalid_or_duplicate_contract(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    _write_linear_policy(path)
    with pytest.raises(ValueError, match="positive"):
        bake_action_clip(path, 0.0)
    bake_action_clip(path, 1.0)
    with pytest.raises(ValueError, match="already contains"):
        bake_action_clip(path, 1.0)
