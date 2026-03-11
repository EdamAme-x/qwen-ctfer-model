#!/usr/bin/env bun

import { existsSync } from "node:fs";
import { resolve } from "node:path";

const argv = process.argv.slice(2);
if (argv.length === 0) {
  console.error("Usage: bun scripts/run_python.js <script.py> [args...]");
  process.exit(2);
}

const repoRoot = process.cwd();
const isWindows = process.platform === "win32";

const envPython = process.env.QWEN_CTFER_PYTHON;
const candidates = [
  envPython,
  isWindows ? resolve(repoRoot, ".venv-run", "Scripts", "python.exe") : resolve(repoRoot, ".venv-run", "bin", "python"),
  isWindows ? resolve(repoRoot, ".venv-win", "Scripts", "python.exe") : resolve(repoRoot, ".venv-win", "bin", "python"),
  isWindows ? resolve(repoRoot, ".venv-local", "Scripts", "python.exe") : resolve(repoRoot, ".venv-local", "bin", "python"),
  isWindows ? resolve(repoRoot, ".venv", "Scripts", "python.exe") : resolve(repoRoot, ".venv", "bin", "python"),
  Bun.which("python"),
  Bun.which("python3"),
].filter(Boolean);

const python = candidates.find((item) => existsSync(String(item)));
if (!python) {
  console.error(
    "No usable Python found. Set QWEN_CTFER_PYTHON or create .venv-run/.venv-win/.venv-local/.venv.",
  );
  process.exit(1);
}

const child = Bun.spawn({
  cmd: [String(python), ...argv],
  cwd: repoRoot,
  stdin: "inherit",
  stdout: "inherit",
  stderr: "inherit",
  env: process.env,
});

const exitCode = await child.exited;
process.exit(exitCode);
