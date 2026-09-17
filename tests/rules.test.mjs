import test from "node:test";
import assert from "node:assert/strict";
import { CENTER, applyMove, emptyBoard, premiumAt, validateMove } from "../src/rules.js";
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
test("standard premium squares score fresh tiles once", () => {
  assert.equal(premiumAt(0, 0), "TW"); assert.equal(premiumAt(1, 5), "TL"); assert.equal(premiumAt(2, 6), "DL"); assert.equal(premiumAt(CENTER, CENTER), "DW");
  const tripleWordBoard=emptyBoard(); applyMove(tripleWordBoard,[{row:0,col:6,letter:"A"}]);
  assert.equal(validateMove(tripleWordBoard,[{row:0,col:7,letter:"T"}],words).score,6);
  const tripleLetterBoard=emptyBoard(); applyMove(tripleLetterBoard,[{row:1,col:4,letter:"A"}]);
  assert.equal(validateMove(tripleLetterBoard,[{row:1,col:5,letter:"T"}],words).score,4);
  const usedDoubleWordBoard=emptyBoard(); applyMove(usedDoubleWordBoard,[{row:1,col:1,letter:"A"}]);
  assert.equal(validateMove(usedDoubleWordBoard,[{row:1,col:2,letter:"T"}],words).score,2);
});
