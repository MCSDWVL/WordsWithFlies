"""GPU bootstrap trainer for the Words with Flies candidate-ranking policy.

It trains the real fixed-connectome architecture against a deterministic teacher.
The browser's legal-move generator remains the final authority on candidates.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FlyWordPolicy

VALUES = {letter: value for letter, value in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10])}

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

class TeacherBatches:
    """Deterministic candidate turns; later game-record shards plug in here unchanged."""
    def __init__(self, words_path: Path, seed: int, candidate_count: int = 64):
        self.rng = random.Random(seed); self.candidate_count = candidate_count
        self.words = [word.strip() for word in words_path.read_text(encoding="utf8").splitlines() if 2 <= len(word.strip()) <= 7]
        if not self.words: raise ValueError("No usable 2-7 letter words")
    def batch(self, size: int):
        state = np.zeros((size, 96), np.float32); candidates = np.zeros((size, self.candidate_count, 48), np.float32)
        target, values = np.zeros(size, np.int64), np.zeros(size, np.float32)
        for row in range(size):
            # Board/rack/bag summary features. Slots 0:26,26:52,52:78 are letter counts.
            state[row, :78] = np.asarray([self.rng.random() for _ in range(78)], np.float32)
            state[row, 78:84] = [self.rng.random(), self.rng.random(), self.rng.uniform(-1,1), self.rng.random(), self.rng.random(), 1.0]
            scores = []
            for col in range(self.candidate_count):
                word = self.rng.choice(self.words); counts = np.zeros(26, np.float32)
                for letter in word: counts[ord(letter) - 65] += 1
                raw = sum(VALUES[letter] for letter in word); premium = self.rng.choice((0, 0, 0, 1, 2))
                candidates[row, col, :26] = counts / 7
                candidates[row, col, 26:34] = [len(word)/7, raw/50, premium/2, self.rng.random(), self.rng.random(), self.rng.random(), self.rng.random(), 0]
                # Teacher favors score, length, premium use and a balanced leave; noise preserves variation.
                score = raw * (1 + .35 * premium) + len(word) * 1.4 + self.rng.uniform(-.35, .35)
                scores.append(score)
            target[row] = int(np.argmax(scores)); values[row] = np.tanh(max(scores) / 45)
        return state, candidates, target, values

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=None); parser.add_argument("--batch-size", type=int, default=None); parser.add_argument("--resume", type=Path); args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent; cfg = json.loads((root / "training/config.json").read_text()); run = cfg["training"]
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"]); torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; use the project .venv CUDA PyTorch build.")
    topology_path = root / "training/artifacts/topology.npz"; topology = np.load(topology_path)
    model = FlyWordPolicy(topology, **cfg["model"]).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=run["learning_rate"], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda"); start = 0; best = float("inf")
    if args.resume:
        saved = torch.load(args.resume, weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = saved["step"]; best = saved.get("best_loss", float("inf"))
    batches = TeacherBatches(root / "public/data/words.txt", cfg["seed"]); steps, batch_size = args.steps or run["imitation_steps"], args.batch_size or run["batch_size"]
    release = root / "training/artifacts/checkpoints"; release.mkdir(parents=True, exist_ok=True)
    for step in range(start + 1, steps + 1):
        state, candidates, target, value = batches.batch(batch_size)
        state = torch.from_numpy(state).cuda(non_blocking=True); candidates = torch.from_numpy(candidates).cuda(non_blocking=True); target = torch.from_numpy(target).cuda(); value = torch.from_numpy(value).cuda(); valid = torch.ones(target.shape[0], candidates.shape[1], dtype=torch.bool, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, predicted_value, _ = model(state, candidates, valid)
            loss = F.cross_entropy(logits, target) + .25 * F.mse_loss(predicted_value, value)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        if step == 1 or step % 100 == 0: print(f"step={step} loss={loss.item():.5f} cuda_mem_mb={torch.cuda.max_memory_allocated() // 1048576}", flush=True)
        if loss.item() < best:
            best = loss.item(); torch.save({"step":step,"best_loss":best,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":cfg,"topology_sha256":sha(topology_path),"lexicon_sha256":sha(root / "public/data/words.txt")}, release / "best-dense.pt")
        if step % run["checkpoint_every"] == 0: torch.save({"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":cfg}, release / f"step-{step}.pt")
    print(f"Saved best checkpoint; final loss {loss.item():.5f}")
if __name__ == "__main__": main()
