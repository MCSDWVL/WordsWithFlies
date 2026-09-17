from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from model import FlyWordPolicy

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, default=Path("training/artifacts/checkpoints/best-dense.pt")); parser.add_argument("--model", type=Path, default=Path("public/models/words-with-flies.onnx")); args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent; saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False); topology = np.load(root / "training/artifacts/topology.npz")
    model = FlyWordPolicy(topology, **saved["config"]["model"]); model.load_state_dict(saved["model"]); model.eval()
    rng = np.random.default_rng(19); state = rng.normal(size=(1,96)).astype(np.float32); candidates = rng.normal(size=(1,64,48)).astype(np.float32); valid = np.ones((1,64), np.bool_); recurrent = rng.normal(size=(1,model.node_count)).astype(np.float32)
    with torch.no_grad(): expected = model(torch.from_numpy(state), torch.from_numpy(candidates), torch.from_numpy(valid), torch.from_numpy(recurrent))
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"]); actual = session.run(None, {"state":state,"candidates":candidates,"valid":valid,"recurrent":recurrent})
    errors = [float(np.max(np.abs(left.numpy() - right))) for left, right in zip(expected, actual)]
    print({"providers":session.get_providers(),"max_abs_error":errors})
    if max(errors) > 2e-5: raise SystemExit("ONNX parity failed")
if __name__ == "__main__": main()
