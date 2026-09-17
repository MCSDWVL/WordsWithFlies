import { CENTER, SIZE, VALUES, validateMove } from "./rules.js";

let words = null;
let loading = null;
let modelSession = null;
let recurrent = null;
async function lexicon() {
  if (words) return words;
  if (!loading) loading = fetch("../public/data/words.txt").then((response) => {
    if (!response.ok) throw new Error("Missing public/data/words.txt — run npm run build:lexicon first.");
    return response.text();
  }).then((text) => new Set(text.trim().split(/\r?\n/)));
  words = await loading; return words;
}
function value(word) { return [...word].reduce((sum, letter) => sum + (VALUES[letter] || 0), 0); }
function canPossiblyMake(word, rack, boardLetters) {
  const counts = new Map(); for (const letter of rack) counts.set(letter, (counts.get(letter) || 0) + 1);
  let blanks = counts.get("?") || 0;
  for (const letter of word) { if (counts.get(letter)) counts.set(letter, counts.get(letter) - 1); else if (boardLetters.has(letter)) continue; else if (blanks--) continue; else return false; }
  return true;
}
function choices(board, rack, dictionary) {
  const occupied = [];
  const boardLetters = new Set();
  board.forEach((row, r) => row.forEach((cell, c) => { if (cell) { occupied.push([r, c, cell.letter]); boardLetters.add(cell.letter); } }));
  const useful = [];
  for (const word of dictionary) if (word.length <= 7 && canPossiblyMake(word, rack, boardLetters)) useful.push(word);
  useful.sort((a, b) => value(b) - value(a) || b.length - a.length);
  const candidates = [];
  const attempt = (word, row, col, horizontal) => {
    const move = [];
    for (let index = 0; index < word.length; index++) {
      const r = row + (horizontal ? 0 : index), c = col + (horizontal ? index : 0);
      if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) return;
      const existing = board[r][c]; if (existing && existing.letter !== word[index]) return;
      if (!existing) move.push({ row: r, col: c, letter: word[index] });
    }
    if (!move.length) return;
    const check = validateMove(board, move, dictionary); if (check.ok) candidates.push({ move, word, score: check.score });
  };
  for (const word of useful.slice(0, 1400)) {
    if (!occupied.length) for (let i = 0; i < word.length; i++) attempt(word, CENTER, CENTER - i, true);
    else for (const [r, c, letter] of occupied) for (let i = 0; i < word.length; i++) if (word[i] === letter) { attempt(word, r, c - i, true); attempt(word, r - i, c, false); }
    if (candidates.length > 700) break;
  }
  return candidates;
}
function brainScores(candidates, board, rack) {
  // Deterministic tiny recurrent readout: replace with the trained ONNX policy later.
  const state = new Float32Array(48); const seed = board.flat().filter(Boolean).length + rack.join("").length * 17;
  for (let step = 0; step < 8; step++) for (let i = 0; i < state.length; i++) state[i] = Math.tanh(state[i] * .71 + Math.sin(seed + i * 1.73 + step) * .34);
  return candidates.map((candidate, index) => candidate.score + candidate.word.length * 1.35 + state[index % state.length] * 2.2 - index * .0001);
}
function exchangeChoice(rack, candidates, bagCount) {
  if (bagCount < 7 || rack.length < 2) return null;
  const bestScore = candidates.reduce((best, candidate) => Math.max(best, candidate.score), 0);
  if (candidates.length && bestScore >= 9) return null;
  // Keep premium consonants; swap surplus vowels and low-value common tiles.
  const utility = { A:2,E:2,I:2,O:2,U:2,L:3,N:3,R:3,S:3,T:3,"?":8 };
  return rack.map((letter, index) => ({ index, value: utility[letter] || (VALUES[letter] || 0) }))
    .sort((left, right) => left.value - right.value).slice(0, Math.min(3, rack.length - 1)).map((entry) => entry.index);
}
function stateFeatures(board, rack) {
  const features = new Float32Array(96); let occupied = 0;
  for (const row of board) for (const cell of row) if (cell) { features[cell.letter.charCodeAt(0) - 65] += 1 / 15; occupied++; }
  for (const letter of rack) features[26 + (letter === "?" ? 26 : letter.charCodeAt(0) - 65)] += 1 / 7;
  features[78] = occupied / 225; features[79] = rack.length / 7; features[83] = 1;
  return features;
}
function candidateFeatures(candidates) {
  const features = new Float32Array(candidates.length * 48);
  candidates.forEach((candidate, index) => {
    const offset = index * 48;
    for (const letter of candidate.word) features[offset + letter.charCodeAt(0) - 65] += 1 / 7;
    const first = candidate.move[0];
    features.set([candidate.word.length / 7, candidate.score / 50, 0, first.row / 14, first.col / 14, 0, 0, 0], offset + 26);
  });
  return features;
}
async function onnxScores(candidates, board, rack) {
  const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.min.mjs");
  if (!modelSession) {
    ort.env.wasm.numThreads = 1;
    modelSession = await ort.InferenceSession.create("../public/models/words-with-flies.onnx", { executionProviders: ["wasm"] });
    recurrent = new Float32Array(2952);
  }
  const count = candidates.length;
  const result = await modelSession.run({
    state: new ort.Tensor("float32", stateFeatures(board, rack), [1, 96]),
    candidates: new ort.Tensor("float32", candidateFeatures(candidates), [1, count, 48]),
    valid: new ort.Tensor("bool", new Uint8Array(count).fill(1), [1, count]),
    recurrent: new ort.Tensor("float32", recurrent, [1, 2952])
  });
  recurrent = result.next_recurrent.data;
  // Copy the complete recurrent state for the neural field. `slice` is
  // intentional: transferring the backing buffer would detach `recurrent`,
  // which is also the model's state for the next fly turn.
  const activity = recurrent.slice();
  return { scores: Array.from(result.policy_logits.data), activity };
}
self.onmessage = async ({ data }) => {
  try {
    const dictionary = await lexicon();
    const candidates = choices(data.board, data.rack, dictionary).slice(0, 64);
    const exchangeIndices = exchangeChoice(data.rack, candidates, data.bagCount || 0);
    if (exchangeIndices?.length) return self.postMessage({ type: "move", exchange: true, exchangeIndices, candidates: candidates.length, model: "rack exchange", activity: [] });
    if (!candidates.length) return self.postMessage({ type: "move", pass: true, candidates: 0, activity: [] });
    let scores, activity = []; let model = "trained ONNX";
    try { const inference = await onnxScores(candidates, data.board, data.rack); scores = inference.scores; activity = inference.activity; }
    catch (error) { scores = brainScores(candidates, data.board, data.rack); model = `fallback: ${error.message || "model load failed"}`; }
    let winner = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[winner]) winner = i;
    self.postMessage({ type: "move", ...candidates[winner], candidates: candidates.length, model, activity });
  } catch (error) { self.postMessage({ type: "error", message: error.message }); }
};
