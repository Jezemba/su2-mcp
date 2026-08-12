"""Tools that execute SU2 solvers."""

from __future__ import annotations

from su2_mcp.config_utils import ensure_force_output
from su2_mcp.su2_runner import (
    SU2Runner,
    build_last_run_metadata,
    parse_config_errors,
    parse_fatal_error,
)
from su2_mcp.tools.session import SESSION_MANAGER, _error


_COEFF_ALIASES = {
    "CL": ("CL", "LIFT"),
    "CD": ("CD", "DRAG"),
    "CMz": ("CMZ", "CM", "MOMENT_Z"),
}


def final_coefficients(workdir: object) -> dict[str, float]:
    """Read the converged force coefficients straight out of the history file.

    ``read_history_csv`` was the ONLY place the framework captured CL/CD, which
    made coupling depend on the caller choosing that particular tool to read
    results with. Measured live (sweep 2026-08-12, run 3/16): SU2 solved cleanly
    in 155 s, the caller reached for ``sample_surface_solution`` instead, hit a
    dead end, and returned -- so the mission flew aviary's DEFAULT drag even
    though the coefficients had been computed and written correctly. Run 1/16 of
    the same sweep coupled purely because the caller happened to pick the other
    tool.

    A solver that has just computed these values should report them, rather than
    leaving them to be fetched by exactly one of several plausible follow-ups.
    Best-effort: never let a reporting problem fail an otherwise good solve.
    """
    import csv as _csv
    from pathlib import Path as _Path

    try:
        wd = _Path(str(workdir))
        candidates = sorted(wd.glob("history*.csv"))
        if not candidates:
            return {}
        with candidates[0].open("r", encoding="utf-8", errors="replace") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            return {}
        # SU2 writes padded, quoted headers -- '       "CD"       '.
        last = {str(k).strip().strip('"').strip().upper(): v for k, v in rows[-1].items()}
        out: dict[str, float] = {}
        for name, aliases in _COEFF_ALIASES.items():
            for alias in aliases:
                if alias in last:
                    try:
                        out[name] = float(str(last[alias]).strip())
                    except (TypeError, ValueError):
                        continue
                    break
        return out
    except Exception:  # pragma: no cover - reporting must never break a solve
        return {}


def _as_int(value: object, default: int = -1) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def run_su2_solver(
    session_id: str,
    solver: str = "SU2_CFD",
    config_override_path: str | None = None,
    max_runtime_seconds: int = 600,
    capture_log_lines: int = 100,
) -> dict[str, object]:
    """Run a SU2 solver process and capture output metadata."""
    try:
        record = SESSION_MANAGER.require(session_id)
        config_path = (
            record.workdir / config_override_path
            if config_override_path
            else record.config_path
        )
        # Guarantee the run will actually WRITE force coefficients before we
        # spend the solve. Without MARKER_MONITORING (and LIFT/DRAG/AERO_COEFF
        # in HISTORY_OUTPUT) SU2 converges happily and emits a history.csv of
        # residuals only -- no CL/CD -- so a downstream mission silently falls
        # back to its default drag polar. Observed in a real MAS-Aviary
        # networked run 2026-08-02. Non-destructive: only missing pieces are
        # added, so a correctly configured session is untouched.
        force_output: dict[str, object] = {}
        try:
            force_output = ensure_force_output(config_path)
        except Exception as exc:  # never block a solve on the guard itself
            force_output = {"error": f"force-output check failed: {exc}"}

        runner = SU2Runner(record.workdir)
        result = runner.run(solver, config_path, max_runtime_seconds, capture_log_lines)
        if "error" not in result:
            metadata = build_last_run_metadata(result)
            SESSION_MANAGER.record_run(session_id, metadata)
        if isinstance(result, dict):
            result["force_output"] = force_output
            if result.get("success"):
                coeffs = final_coefficients(record.workdir)
                if coeffs:
                    result["final_coefficients"] = coeffs
                else:
                    result["final_coefficients_note"] = (
                        "The solve succeeded but no CL/CD could be read from the "
                        "history file. Check that MARKER_MONITORING names a real "
                        "wall marker and HISTORY_OUTPUT includes AERO_COEFF."
                    )
            # Surface SU2's own config complaints as structured, actionable
            # items instead of leaving them buried in the log_tail blob. SU2
            # names the offending option AND the correct spelling, so a caller
            # can fix it in one step rather than re-guessing.
            cfg_errors = parse_config_errors(str(result.get("log_tail", "")))
            if cfg_errors:
                result["config_errors"] = cfg_errors
                result["hint"] = (
                    "SU2 rejected the configuration and did NOT solve, so there is "
                    "no history and no CL/CD. Fix the options listed in "
                    "config_errors (use the did_you_mean spellings) via "
                    "update_config_entries, then re-run."
                )
            elif not result.get("success"):
                # Any OTHER fatal exit. SU2 words these well but prints them
                # mid-banner, so promote its own text rather than leaving the
                # caller to find line 35 of 47.
                fatal = parse_fatal_error(str(result.get("log_tail", "")))
                if fatal:
                    result["solver_error"] = fatal["message"]
                    result["solver_error_location"] = fatal["location"]
                    result["hint"] = (
                        f"SU2 exited without solving: {fatal['message']} There is no "
                        "history and no CL/CD. Resolve that condition before re-running; "
                        "re-issuing the same call unchanged will fail identically."
                    )
        return result
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to run solver", details=str(exc))


def generate_deformed_mesh(
    session_id: str,
    def_config_path: str | None = None,
    output_mesh_name: str = "mesh_def.su2",
    max_runtime_seconds: int = 600,
) -> dict[str, object]:
    """Run SU2_DEF to create a deformed mesh."""
    try:
        record = SESSION_MANAGER.require(session_id)
        config_path = (
            record.workdir / def_config_path if def_config_path else record.config_path
        )
        runner = SU2Runner(record.workdir)
        result = runner.run(
            "SU2_DEF", config_path, max_runtime_seconds, capture_log_lines=200
        )
        success = bool(result.get("success"))
        result_payload: dict[str, object] = {
            "success": success,
            "exit_code": _as_int(result.get("exit_code", -1)),
            "runtime_seconds": _as_float(result.get("runtime_seconds", 0.0)),
            "log_tail": str(result.get("log_tail", "")),
            "deformed_mesh_path": str(record.workdir / output_mesh_name)
            if success
            else None,
        }
        if "error" in result:
            result_payload["error"] = result["error"]
        return result_payload
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to generate deformed mesh", details=str(exc))


__all__ = ["run_su2_solver", "generate_deformed_mesh"]
