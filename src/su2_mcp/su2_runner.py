"""Helpers for invoking SU2 binaries safely."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from su2_mcp.session_manager import LastRunMetadata

# SU2 rejects a bad config before solving and says EXACTLY what is wrong,
# including the correct spelling:
#
#   Line 9 MARKER_BODY: invalid option name. Check current SU2 options in
#   config_template.cfg.
#   Did you mean MARKER_EMISSIVITY?
#   TIME_DISCRE_FLOW: invalid option value IMPLICIT.
#   Did you mean, ADER_DG, CLASSICAL_RK4_EXPLICIT, EULER_EXPLICIT, EULER_IMPLICIT, ...?
#
# That guidance was buried inside the `log_tail` prose blob, so agents could not
# act on it. Measured across two MAS-Aviary sweeps: SU2 failed 42/46 and 20/25
# solves, roughly two thirds of them on invalid options — which is why the aero
# coupling almost never happened. The mistakes are near-misses (GREEN-GAUSS vs
# GREEN_GAUSS, IMPLICIT vs EULER_IMPLICIT), so a structured, actionable error
# lets the caller self-correct in one step.
_BAD_NAME_RE = re.compile(r"Line\s+(\d+)\s+(\S+?):\s*invalid option name", re.I)
_BAD_VALUE_RE = re.compile(r"(\S+?):\s*invalid option value\s*(\S*)", re.I)
_SUGGEST_RE = re.compile(r"Did you mean,?\s*([^?]*)\?", re.I)

# A second, nastier class: a REQUIRED option is missing, and SU2 8.3 names it by
# its DEPRECATED v6 spelling. Observed live:
#
#   "PHYSICAL_PROBLEM must be set in the configuration file"
#
# but PHYSICAL_PROBLEM is exactly what SU2 8.3 rejects as "invalid option name"
# — the modern spelling is SOLVER. An agent that follows the error literally
# loops: set PHYSICAL_PROBLEM -> invalid option name -> set it again. We
# translate through the existing deprecation map so the caller is told the name
# SU2 will actually accept.
_MISSING_RE = re.compile(r"(\S+?)\s+must be set in the configuration file", re.I)


def parse_config_errors(log_text: str) -> list[dict[str, object]]:
    """Extract SU2's config-parsing complaints as structured, actionable items.

    Each entry carries the offending option, whether the NAME or the VALUE was
    rejected, and SU2's own suggested replacements. Returns [] when the log
    holds no config errors (e.g. a solve that failed for a different reason).
    """
    if not log_text:
        return []
    # SU2 writes these as literal "\n" inside the JSON string as well as real
    # newlines depending on how the log was captured; normalise both.
    text = log_text.replace("\\n", "\n")
    lines = [ln.strip() for ln in text.splitlines()]

    errors: list[dict[str, object]] = []
    for i, line in enumerate(lines):
        entry: dict[str, object] | None = None
        m = _BAD_NAME_RE.search(line)
        if m:
            entry = {
                "option": m.group(2),
                "problem": "invalid option name",
                "line": int(m.group(1)),
            }
        else:
            m = _BAD_VALUE_RE.search(line)
            if m:
                entry = {
                    "option": m.group(1),
                    "problem": "invalid option value",
                    "value": m.group(2).rstrip(".") or None,
                }
            else:
                m = _MISSING_RE.search(line)
                if m:
                    from su2_mcp.config_utils import DEPRECATED_OPTIONS

                    named = m.group(1)
                    modern = DEPRECATED_OPTIONS.get(named.upper())
                    entry = {"option": named, "problem": "required option missing"}
                    if modern:
                        entry["did_you_mean"] = [modern]
                        entry["note"] = (
                            f"SU2 names this by its deprecated spelling. Set "
                            f"{modern} — {named} is rejected as an invalid option "
                            f"name in this SU2 version."
                        )
        if entry is None:
            continue
        # SU2 puts "Did you mean ...?" on this line or the next one.
        for probe in (line, lines[i + 1] if i + 1 < len(lines) else ""):
            s = _SUGGEST_RE.search(probe)
            if s:
                entry["did_you_mean"] = [
                    tok.strip() for tok in s.group(1).split(",") if tok.strip()
                ]
                break
        errors.append(entry)
    return errors


class SU2Runner:
    """Run SU2 commands with subprocess while capturing metadata."""

    def __init__(self, workdir: Path) -> None:
        """Create a runner bound to a specific working directory."""
        self.workdir = workdir

    def run(
        self,
        solver: str,
        config_path: Path,
        max_runtime_seconds: int,
        capture_log_lines: int,
    ) -> dict[str, object]:
        """Execute a SU2 solver and return structured metadata."""
        start = time.time()
        try:
            process = subprocess.run(
                [solver, str(config_path)],
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max_runtime_seconds,
                check=False,
                text=True,
            )
            output_text = process.stdout or ""
            runtime = time.time() - start
            tail_lines = "\n".join(output_text.splitlines()[-capture_log_lines:])
            residual_history = self._parse_history_files()
            return {
                "success": process.returncode == 0,
                "solver": solver,
                "config_used": str(config_path),
                "exit_code": int(process.returncode),
                "runtime_seconds": runtime,
                "log_tail": tail_lines,
                "residual_history": residual_history,
            }
        except subprocess.TimeoutExpired as exc:
            runtime = time.time() - start
            return {
                "success": False,
                "solver": solver,
                "config_used": str(config_path),
                "exit_code": -1,
                "runtime_seconds": runtime,
                "log_tail": "Process timed out",
                "residual_history": None,
                "error": {
                    "type": "timeout",
                    "message": "SU2 solver execution timed out",
                    "details": str(exc),
                },
            }
        except FileNotFoundError as exc:
            runtime = time.time() - start
            return {
                "success": False,
                "solver": solver,
                "config_used": str(config_path),
                "exit_code": -1,
                "runtime_seconds": runtime,
                "log_tail": "",
                "residual_history": None,
                "error": {
                    "type": "missing_binary",
                    "message": f"Solver binary '{solver}' not found",
                    "details": str(exc),
                },
            }

    def _parse_history_files(self) -> list[dict[str, object]] | None:
        for candidate in (self.workdir / "history.csv", self.workdir / "history.dat"):
            if candidate.exists():
                return self._read_history(candidate)
        return None

    def _read_history(self, path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
                for line in handle:
                    values = [value.strip() for value in line.split(",")]
                    if len(values) != len(header):
                        continue
                    entry: dict[str, object] = {}
                    for key, value in zip(header, values, strict=False):
                        try:
                            entry[key] = float(value)
                        except ValueError:
                            entry[key] = value
                    rows.append(entry)
        except FileNotFoundError:
            return []
        return rows


def build_last_run_metadata(result: Mapping[str, object]) -> LastRunMetadata:
    """Convert a solver result payload into `LastRunMetadata`."""
    exit_code_raw = result.get("exit_code", -1)
    runtime_raw = result.get("runtime_seconds", 0.0)

    exit_code = (
        int(exit_code_raw) if isinstance(exit_code_raw, (int, float, str)) else -1
    )
    runtime_seconds = (
        float(runtime_raw) if isinstance(runtime_raw, (int, float, str)) else 0.0
    )

    return LastRunMetadata(
        solver=str(result.get("solver", "")),
        config_used=str(result.get("config_used", "")),
        exit_code=exit_code,
        runtime_seconds=runtime_seconds,
        log_tail=str(result.get("log_tail", "")),
    )
