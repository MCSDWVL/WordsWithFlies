"""Build a compact adult FlyWire topology from the pinned public source."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

REVISION = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
BASE = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{REVISION}/"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fetch(cache: Path, filename: str) -> Path:
    path = cache / filename
    if not path.exists():
        print(f"Downloading {filename}", flush=True)
        urllib.request.urlretrieve(BASE + filename, path)
    return path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("training/cache/upstream"))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/topology.npz"))
    parser.add_argument("--neurons", type=int, default=2952)
    parser.add_argument("--max-edges", type=int, default=110000)
    args = parser.parse_args(); args.cache.mkdir(parents=True, exist_ok=True); args.output.parent.mkdir(parents=True, exist_ok=True)
    parquet, license_file = fetch(args.cache, "Connectivity_783.parquet"), fetch(args.cache, "LICENSE")
    edge = pd.read_parquet(parquet, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = edge["Presynaptic_Index"].to_numpy(np.int32); post = edge["Postsynaptic_Index"].to_numpy(np.int32)
    weight = edge["Excitatory x Connectivity"].to_numpy(np.float32); count = int(max(pre.max(), post.max())) + 1
    strength = np.bincount(pre, weights=np.abs(weight), minlength=count) + np.bincount(post, weights=np.abs(weight), minlength=count)
    # lexsort makes ties stable by original neuron index.
    nodes = np.lexsort((np.arange(count), -strength))[:args.neurons].astype(np.int32)
    remap = np.full(count, -1, np.int32); remap[nodes] = np.arange(len(nodes), dtype=np.int32)
    mask = (remap[pre] >= 0) & (remap[post] >= 0); compact_pre, compact_post, compact_weight = remap[pre[mask]], remap[post[mask]], weight[mask]
    edge_order = np.lexsort((compact_post, compact_pre, -np.abs(compact_weight)))[:args.max_edges]
    compact_pre, compact_post, compact_weight = compact_pre[edge_order], compact_post[edge_order], compact_weight[edge_order]
    degree = np.bincount(compact_pre, weights=np.abs(compact_weight), minlength=len(nodes)) + np.bincount(compact_post, weights=np.abs(compact_weight), minlength=len(nodes))
    ranked = np.lexsort((np.arange(len(nodes)), -degree)); sensory, motor = ranked[:256].astype(np.int32), ranked[256:512].astype(np.int32)
    np.savez_compressed(args.output, source_nodes=nodes, src=compact_pre, dst=compact_post, initial_weight=compact_weight, sensory=sensory, motor=motor)
    manifest = {"revision": REVISION, "source": BASE, "source_sha256": {"Connectivity_783.parquet": sha(parquet), "LICENSE": sha(license_file)}, "source_neurons": count, "neurons": len(nodes), "edges": len(compact_pre), "topology_sha256": sha(args.output), "license": "MIT (copied from upstream source cache)"}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    (args.output.parent / "UPSTREAM-MIT-LICENSE.txt").write_bytes(license_file.read_bytes())
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
