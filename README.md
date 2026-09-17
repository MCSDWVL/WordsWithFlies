# Words with Flies

A static, browser-only word game where you play a tile-board game against a visible fruit-fly-inspired decision system.

## Run it

```powershell
npm run build:lexicon
npm run serve
```

Open the local address printed by `serve`. The first command takes the supplied `T:/OtherProjects/Lexicon/dictionary.txt`, normalizes its A–Z words, and writes the deployable assets under `public/data/`. Pass a different source list with `node scripts/build-lexicon.mjs <path-to-dictionary.txt>`.

The site has no backend: rule validation, the lexicon, game state, and fly move selection all run locally. Deploy the repository contents, including `public/data/words.txt`, through GitHub Pages or another static host.

## Current fly and training replacement

`src/fly-worker.js` deliberately isolates the fly move scorer behind `brainScores`. It currently supplies a deterministic, animated fallback so the entire game is playable before model training. `training/README.md` specifies the replacement contract: train a separately licensed small fruit-fly connectome model offline, export it to ONNX, then replace that scorer with browser-side ONNX inference while retaining the same legal-candidate and rules path.

## Train the browser model

The project-local `.venv` is CUDA-enabled and was verified against an RTX 3080. The first exported model is a real trained compact adult-FlyWire-connectome policy, but it is a bootstrap teacher-ranking model rather than a quality-complete opponent.

```powershell
.\.venv\Scripts\python.exe training\prepare_topology.py
.\.venv\Scripts\python.exe training\train.py --steps 10000 --batch-size 8
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe training\export_onnx.py
.\.venv\Scripts\python.exe training\check_onnx.py
```

`prepare_topology.py` caches the pinned upstream connectivity source outside deployable assets and creates a deterministic 2,952-neuron, 110,000-edge topology. The exported model and its external `.onnx.data` file under `public/models/` must stay together when deploying. The browser Worker attempts that ONNX policy first and identifies the visible fallback if the runtime or model cannot load.
