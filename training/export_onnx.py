from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from model import FlyWordPolicy

class ExportPolicy(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, state, candidates, valid, recurrent):
        logits, value, next_state = self.model(state, candidates, valid.bool(), recurrent)
        return logits, value, next_state

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, default=Path("training/artifacts/checkpoints/best-dense.pt")); parser.add_argument("--output", type=Path, default=Path("public/models/words-with-flies.onnx")); args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent; saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False); topology = np.load(root / "training/artifacts/topology.npz")
    model = FlyWordPolicy(topology, **saved["config"]["model"]); model.load_state_dict(saved["model"]); model.eval(); wrapped = ExportPolicy(model)
    args.output.parent.mkdir(parents=True, exist_ok=True); state = torch.zeros(1, 96); candidates = torch.zeros(1, 64, 48); valid = torch.ones(1, 64, dtype=torch.bool); recurrent = torch.zeros(1, model.node_count)
    # ORT Web's WASM backend does not mount PyTorch's separate `.onnx.data`
    # file. Keep all 35 MiB of learned graph weights inside one portable file.
    torch.onnx.export(wrapped, (state, candidates, valid, recurrent), args.output, input_names=["state","candidates","valid","recurrent"], output_names=["policy_logits","value","next_recurrent"], opset_version=18, dynamo=True, external_data=False)
    args.output.with_suffix(".json").write_text(json.dumps({"checkpoint":str(args.checkpoint),"topology_sha256":saved.get("topology_sha256"),"lexicon_sha256":saved.get("lexicon_sha256"),"state_features":96,"candidate_features":48,"recurrent_nodes":model.node_count}, indent=2)+"\n")
    print(args.output)
if __name__ == "__main__": main()
