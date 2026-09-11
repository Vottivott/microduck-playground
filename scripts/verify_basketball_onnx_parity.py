"""Check normalized recurrent basketball ONNX against its PyTorch checkpoint.

Checks a sequence with carried state and an explicit policy-reset boundary.
Export through scripts/export.py first; this script never converts checkpoints.
"""
import argparse
import json

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict

from mjlab_microduck.basketball_distillation import load_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("onnx")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs = TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1])
    actor = load_actor(checkpoint, obs)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.onnx, sess_options=options, providers=["CPUExecutionProvider"])
    assert [x.name for x in session.get_inputs()] == ["obs", "h_in", "c_in"]
    hidden = np.zeros((actor.rnn.rnn.num_layers, 1, actor.rnn.rnn.hidden_size), dtype=np.float32)
    cell = hidden.copy()
    error = 0.0
    for step in range(40):
        if step == 20:
            actor.reset()
            hidden.fill(0)
            cell.fill(0)
        x = torch.randn(1, 61) * 0.2
        x[:, -10:] = 0  # head/body command padding; twist remains live
        with torch.inference_mode():
            expected = actor(TensorDict({"actor": x}, batch_size=[1])).numpy()
        actual, hidden, cell = session.run(None, {"obs": x.numpy(), "h_in": hidden, "c_in": cell})
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
        error = max(error, float(np.max(np.abs(actual - expected))))
    print("BASKETBALL_ONNX_PARITY", json.dumps({"steps": 40, "reset_at": 20, "max_abs_error": error}))


if __name__ == "__main__":
    main()
