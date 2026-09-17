import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const source = process.argv[2] || "T:/OtherProjects/Lexicon/dictionary.txt";
const destination = resolve("public/data/words.txt");
if (!existsSync(source)) throw new Error(`Lexicon not found: ${source}`);
const words = [...new Set(readFileSync(source, "utf8").split(/\r?\n/)
  .map((word) => word.trim().toUpperCase())
  .filter((word) => /^[A-Z]{2,15}$/.test(word)))].sort();
mkdirSync(dirname(destination), { recursive: true });
const text = `${words.join("\n")}\n`;
writeFileSync(destination, text);
writeFileSync(resolve("public/data/lexicon-manifest.json"), JSON.stringify({
  source: source.replaceAll("\\", "/"), wordCount: words.length,
  sha256: createHash("sha256").update(text).digest("hex")
}, null, 2) + "\n");
console.log(`Wrote ${words.length} words to ${destination}`);
