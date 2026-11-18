"""Did an SU2 solve converge, and did a fixed-CL solve reach its lift target? (Avion M10)

run_su2_solver reported ``success = (exit code 0)`` and the last history row's CL/CD. SU2 exits 0 when it
stops at the iteration limit too, so an unconverged or untrimmed solve went out looking like a result.
l20_final S_GR 12: the flown solve stopped at 499 of 500 iterations with rms[Rho] -4.23 against the -6
target (CL 0.989, CD 0.182), and it was reported and flown like any other.

This module answers two questions in plain words, from what SU2 itself wrote:

* converged: SU2's own exit verdict ("All convergence criteria satisfied." or "Maximum number of
  iterations reached (ITER = n)") when the log is available; otherwise the final residual of the
  CONV_FIELD column against CONV_RESIDUAL_MINVAL, and the iteration count against ITER.
* trimmed (FIXED_CL_MODE= YES only): the last CL against TARGET_CL, within TRIM_TOLERANCE (relative).

Nothing here guesses an SU2 default: a setting the config does not state is reported as not stated, and
a question that cannot be answered is answered ``None`` with the reason.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

# |CL / TARGET_CL - 1| above this is "not trimmed". SU2's own fixed-CL test is |CL - TARGET_CL| <
# CONV_CAUCHY_EPS / 2 = 0.005 with configure_from_cpacs' setting, i.e. ~1% of a cruise CL near 0.55.
TRIM_TOLERANCE = 0.01

# CONV_FIELD names an SU2 screen/history field; these are the residual ones and their history columns.
_RESIDUAL_COLUMN = {
    "RMS_DENSITY": "rms[Rho]",
    "RMS_MOMENTUM-X": "rms[RhoU]",
    "RMS_MOMENTUM-Y": "rms[RhoV]",
    "RMS_MOMENTUM-Z": "rms[RhoW]",
    "RMS_ENERGY": "rms[RhoE]",
}

_CONVERGED_RE = re.compile(r"All convergence criteria satisfied")
_LIMIT_RE = re.compile(r"Maximum number of (?:time )?iterations reached \(\s*(\w+)\s*=\s*(\d+)")


def _clean(name: object) -> str:
    return str(name).strip().strip('"').strip()


def history_file(workdir: object) -> Path | None:
    """The history file run_su2_solver reads its coefficients from (same choice as final_coefficients)."""
    candidates = sorted(Path(str(workdir)).glob("history*.csv"))
    return candidates[0] if candidates else None


def read_history_rows(path: Path | None) -> list[dict[str, str]]:
    """Every row of a history CSV, with SU2's padded, quoted headers cleaned. [] when unreadable."""
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return [{_clean(k): v for k, v in row.items() if k is not None} for row in csv.DictReader(fh)]
    except OSError:
        return []


def solver_exit_message(log_text: str | None) -> str | None:
    """SU2's own convergence verdict line from a solver log, or None when the log has neither."""
    if not log_text:
        return None
    m = _LIMIT_RE.search(log_text)
    if m:
        return f"Maximum number of iterations reached ({m.group(1)} = {m.group(2)})."
    if _CONVERGED_RE.search(log_text):
        return "All convergence criteria satisfied."
    return None


def _number(value: object) -> float | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_value(row: dict[str, str], *names: str) -> float | None:
    upper = {k.upper(): v for k, v in row.items()}
    for name in names:
        if name.upper() in upper:
            return _number(upper[name.upper()])
    return None


def _fmt(x: float | None, digits: int = 5) -> str:
    return "unknown" if x is None else f"{x:.{digits}g}"


def assess(config: dict[str, Any], rows: list[dict[str, str]], log_text: str | None = None) -> dict[str, Any]:
    """Convergence and trim of one solve. ``config`` is the parsed config the solve ran with."""
    cfg = {str(k).upper(): v for k, v in (config or {}).items()}
    last = rows[-1] if rows else {}

    field = cfg.get("CONV_FIELD")
    if isinstance(field, (list, tuple)):
        field = field[0] if field else None
    field = str(field).strip().upper() if field is not None else None
    column = _RESIDUAL_COLUMN.get(field) if field else None
    final_residual = _row_value(last, column) if column else None
    target_residual = _number(cfg.get("CONV_RESIDUAL_MINVAL"))
    inner = _row_value(last, "Inner_Iter")
    iterations = int(inner) + 1 if inner is not None else None
    max_iterations = _number(cfg.get("ITER"))
    max_iterations = int(max_iterations) if max_iterations is not None else None

    exit_message = solver_exit_message(log_text)
    if exit_message is not None:
        converged: bool | None = exit_message.startswith("All convergence")
        basis = "SU2's exit message"
    elif not rows:
        converged, basis = None, "no history rows"
    elif final_residual is not None and target_residual is not None:
        if final_residual <= target_residual:
            converged, basis = True, f"final {column} against CONV_RESIDUAL_MINVAL"
        elif iterations is not None and max_iterations is not None and iterations >= max_iterations - 1:
            converged, basis = False, f"final {column} against CONV_RESIDUAL_MINVAL at the iteration limit"
        else:
            converged, basis = None, (f"{column} above CONV_RESIDUAL_MINVAL before the iteration limit and no "
                                      "solver exit message: the reason SU2 stopped is unknown")
    else:
        missing = [n for n, v in (("CONV_FIELD residual column", column), ("CONV_RESIDUAL_MINVAL",
                                  target_residual)) if v is None]
        converged, basis = None, f"not stated in the config or history: {', '.join(missing)}"

    residual_name = column or field or "the residual"
    if converged is True:
        conv_text = (f"Converged ({basis}): {residual_name} {_fmt(final_residual, 4)} against the target "
                     f"{_fmt(target_residual, 4)}, after {iterations if iterations is not None else 'unknown'} "
                     f"of {max_iterations if max_iterations is not None else 'unknown'} iterations.")
    elif converged is False:
        conv_text = (f"NOT converged: SU2 stopped at the iteration limit ({iterations if iterations is not None else '?'} "
                     f"of {max_iterations if max_iterations is not None else '?'} iterations) with {residual_name} at "
                     f"{_fmt(final_residual, 4)}; the target is {_fmt(target_residual, 4)}. The last iteration's "
                     "CL and CD are not a converged solution.")
    else:
        conv_text = f"Convergence could not be checked ({basis})."

    trim: dict[str, Any] | None = None
    fixed = str(cfg.get("FIXED_CL_MODE", "")).strip().upper() == "YES"
    achieved = _row_value(last, "CL", "LIFT")
    if fixed:
        target = _number(cfg.get("TARGET_CL"))
        on_target: bool | None = None
        rel: float | None = None
        if target is not None and achieved is not None:
            rel = (achieved / target - 1.0) if target != 0 else None
            on_target = abs(rel) <= TRIM_TOLERANCE if rel is not None else abs(achieved) <= 0.005
        trim = {"fixed_cl_mode": True, "target_cl": target, "achieved_cl": achieved,
                "relative_error": rel, "tolerance": TRIM_TOLERANCE, "on_target": on_target}
        if on_target is True:
            trim_text = f"Trimmed: CL {_fmt(achieved)} is on the target {_fmt(target)} ({rel:+.2%})." if rel is not None \
                else f"Trimmed: CL {_fmt(achieved)} is on the target {_fmt(target)}."
        elif on_target is False:
            trim_text = (f"NOT trimmed: CL {_fmt(achieved)} against the target {_fmt(target)}"
                         + (f" ({rel:+.1%}; tolerance {TRIM_TOLERANCE:.0%})" if rel is not None else "")
                         + ". The drag is not the drag at the target lift.")
        else:
            trim_text = ("Trim could not be checked: "
                         + ("TARGET_CL is not stated in the config." if target is None else "no CL in the history."))
    else:
        trim_text = ("Fixed-angle solve (FIXED_CL_MODE is not YES): no lift target; CL is whatever the configured "
                     "AOA produces.")

    return {
        "converged": converged,
        "basis": basis,
        "solver_exit_message": exit_message,
        "residual_field": residual_name,
        "final_residual": final_residual,
        "target_residual": target_residual,
        "iterations": iterations,
        "max_iterations": max_iterations,
        "trim": trim,
        "trimmed": (trim or {}).get("on_target") if fixed else None,
        "summary": f"{conv_text} {trim_text}",
    }


def assess_solve(config_path: Path, workdir: object, log_text: str | None) -> dict[str, Any]:
    """assess() for a solve that just ran with ``config_path`` in ``workdir``."""
    from su2_mcp.config_utils import parse_config_file

    try:
        config = parse_config_file(Path(config_path)) if Path(config_path).exists() else {}
    except Exception:  # noqa: BLE001 - an unreadable config is reported, not raised
        config = {}
    return assess(config, read_history_rows(history_file(workdir)), log_text)


__all__ = ["TRIM_TOLERANCE", "assess", "assess_solve", "history_file", "read_history_rows",
           "solver_exit_message"]
