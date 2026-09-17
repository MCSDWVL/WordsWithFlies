export const SIZE = 15;
export const CENTER = 7;
export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
export const VALUES = Object.freeze({A:1,B:3,C:3,D:2,E:1,F:4,G:2,H:4,I:1,J:8,K:5,L:1,M:3,N:1,O:1,P:3,Q:10,R:1,S:1,T:1,U:1,V:4,W:4,X:8,Y:4,Z:10});
export const DISTRIBUTION = Object.freeze({A:9,B:2,C:2,D:4,E:12,F:2,G:3,H:2,I:9,J:1,K:1,L:4,M:2,N:6,O:8,P:2,Q:1,R:6,S:4,T:6,U:4,V:2,W:2,X:1,Y:2,Z:1,"?":2});

export function emptyBoard() { return Array.from({ length: SIZE }, () => Array(SIZE).fill(null)); }
export function key(row, col) { return `${row}:${col}`; }
export function tileValue(letter) { return letter === "?" ? 0 : VALUES[letter] || 0; }
export function draw(bag, count) { return bag.splice(Math.max(0, bag.length - count), count); }
export function makeBag(random = Math.random) { const bag = []; for (const [letter, count] of Object.entries(DISTRIBUTION)) bag.push(...Array(count).fill(letter)); for (let i = bag.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [bag[i], bag[j]] = [bag[j], bag[i]]; } return bag; }

function inBounds(row, col) { return row >= 0 && row < SIZE && col >= 0 && col < SIZE; }
function occupied(board, row, col) { return inBounds(row, col) && board[row][col]; }
function wordAt(board, placements, row, col, dr, dc) {
  while (occupied(board, row - dr, col - dc) || placements.get(key(row - dr, col - dc))) { row -= dr; col -= dc; }
  const cells = [];
  while (inBounds(row, col) && (occupied(board, row, col) || placements.get(key(row, col)))) {
    const placed = placements.get(key(row, col)); cells.push({ row, col, letter: placed?.letter || board[row][col].letter, fresh: !!placed }); row += dr; col += dc;
  }
  return cells;
}
function scoreWord(cells) { return cells.reduce((sum, cell) => sum + tileValue(cell.letter), 0); }

export function validateMove(board, move, words) {
  if (!move?.length) return { ok: false, message: "Place at least one tile." };
  const placements = new Map();
  for (const tile of move) {
    if (!inBounds(tile.row, tile.col) || occupied(board, tile.row, tile.col) || placements.has(key(tile.row, tile.col))) return { ok: false, message: "That square is unavailable." };
    placements.set(key(tile.row, tile.col), tile);
  }
  const rows = new Set(move.map((p) => p.row)), cols = new Set(move.map((p) => p.col));
  if (rows.size > 1 && cols.size > 1) return { ok: false, message: "Tiles must stay in one row or column." };
  const horizontal = rows.size === 1;
  const [dr, dc] = horizontal ? [0, 1] : [1, 0];
  const main = wordAt(board, placements, move[0].row, move[0].col, dr, dc);
  if (main.filter((cell) => cell.fresh).length !== move.length) return { ok: false, message: "Tiles must be contiguous." };
  const isFirst = !board.flat().some(Boolean);
  if (isFirst && !placements.has(key(CENTER, CENTER))) return { ok: false, message: "First word must cover the star." };
  if (!isFirst && !move.some((p) => occupied(board, p.row - 1, p.col) || occupied(board, p.row + 1, p.col) || occupied(board, p.row, p.col - 1) || occupied(board, p.row, p.col + 1))) return { ok: false, message: "New tiles must touch the board." };
  const made = [main];
  for (const placement of move) { const cross = wordAt(board, placements, placement.row, placement.col, dc, dr); if (cross.length > 1) made.push(cross); }
  for (const cells of made) { const word = cells.map((cell) => cell.letter).join(""); if (word.length < 2 || !words.has(word)) return { ok: false, message: `“${word}” is not in this lexicon.` }; }
  return { ok: true, words: made.map((cells) => cells.map((cell) => cell.letter).join("")), score: made.reduce((sum, cells) => sum + scoreWord(cells), 0) + (move.length === 7 ? 50 : 0) };
}

export function applyMove(board, move) { for (const tile of move) board[tile.row][tile.col] = { letter: tile.letter, blank: !!tile.blank }; }
