# Words with Flies training

The browser prototype currently uses a deterministic connectome-inspired fallback in `src/fly-worker.js`. Replace only its `brainScores` function with the exported ONNX policy after the pipeline below produces a verified checkpoint.

1. Build a versioned lexicon trie from `public/data/words.txt` and record its SHA-256 hash.
2. Generate complete legal games with the identical rules engine used by the web client. For each turn, retain at most 256 legal placements: high immediate score, rack-balance, board-coverage, defensive, exchange, and pass candidates.
3. Label candidates with a reference word-game solver/heuristic's ranking and final-game value. Train a small, licensed fruit-fly connectome recurrent model with board/rack/bag features, a candidate-ranking policy head, and a win/value head.
4. Train by supervised imitation first, then self-play against random, heuristic, and frozen earlier checkpoints. Preserve seed, lexicon hash, topology hash, candidate policy, and evaluation set in each checkpoint manifest.
5. Export the frozen checkpoint to ONNX; compare state, logits, masked choice, and value against Python fixtures. The web Worker must reject a model whose manifest hashes do not match.

Launch gates: 100% legal selections after masking, sensible completion of seeded games, a positive score margin over random play, varied openings, and acceptable browser move time on mobile and desktop.
