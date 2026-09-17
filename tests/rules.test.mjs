import test from "node:test";
import assert from "node:assert/strict";
import { CENTER, applyMove, emptyBoard, validateMove } from "../src/rules.js";
const words = new Set(["AT", "ATE", "TEA"]);
test("opening play must cover center and be a lexicon word", () => {
  const board = emptyBoard();
  assert.equal(validateMove(board, [{ row: CENTER, col: CENTER, letter: "A" }, { row: CENTER, col: CENTER + 1, letter: "T" }], words).ok, true);
  assert.equal(validateMove(board, [{ row: 0, col: 0, letter: "A" }, { row: 0, col: 1, letter: "T" }], words).ok, false);
});
test("a later move must touch and form valid cross words", () => {
  const board = emptyBoard(); applyMove(board, [{ row: CENTER, col: CENTER, letter: "A" }, { row: CENTER, col: CENTER + 1, letter: "T" }]);
  assert.equal(validateMove(board, [{ row: CENTER - 2, col: CENTER, letter: "T" }, { row: CENTER - 1, col: CENTER, letter: "E" }], words).ok, true);
  assert.equal(validateMove(board, [{ row: 1, col: 1, letter: "A" }, { row: 1, col: 2, letter: "T" }], words).ok, false);
});
