"""Session-oriented MCP tools."""

from __future__ import annotations

from su2_mcp.session_manager import SessionManager

SESSION_MANAGER = SessionManager()


def _error(
    message: str, error_type: str = "runtime_error", details: object | None = None
) -> dict[str, object]:
    """Standardized error response payload."""
    return {"error": {"type": error_type, "message": message, "details": details}}


def create_su2_session(
    base_name: str | None = None,
    initial_config: str | None = None,
    initial_mesh: str | None = None,
    mesh_file_name: str = "mesh.su2",
    cpacs_file_path: str | None = None,
    overrides: dict | None = None,
    ref_area: float | None = None,
    ref_length: float | None = None,
) -> dict[str, object]:
    """Create a new SU2 session, configured from CPACS when a path is given.

    A session created without ``cpacs_file_path`` starts from a minimal config
    that CANNOT solve: SU2 rejects it for absent markers, solver and numerics.
    Recovering from that requires a second call to ``configure_from_cpacs``, and
    callers do not reliably make it. Measured live (sweep 2026-08-12, run 5/16):
    the tool was advertised to the agent, the failing responses named it five
    separate times, and the next action after each was to end the turn -- zero
    ``configure_from_cpacs`` calls across 6 sessions and 6 solves, and the run
    finished with no mission output at all.

    Supplying the CPACS path here makes the session usable on creation, so no
    follow-up call is needed. ``configure_from_cpacs`` remains available for
    reconfiguring an existing session.
    """
    try:
        record = SESSION_MANAGER.create_session(
            base_name=base_name,
            initial_config=initial_config,
            initial_mesh=initial_mesh,
            mesh_file_name=mesh_file_name,
        )
        result: dict[str, object] = {
            "session_id": record.session_id,
            "workdir": str(record.workdir),
            "config_path": str(record.config_path),
            "mesh_path": str(record.mesh_path) if record.mesh_path else None,
        }

        if cpacs_file_path:
            from su2_mcp.tools.config_tools import configure_from_cpacs

            configured = configure_from_cpacs(
                record.session_id,
                cpacs_file_path,
                overrides=overrides,
                mesh_file_name=mesh_file_name,
                ref_area=ref_area,
                ref_length=ref_length,
            )
            if "error" in configured:
                # The session exists and is usable; only the auto-config failed.
                # Say which, so the caller does not discard a good session.
                result["configured"] = False
                result["configure_error"] = configured["error"]
                result["warning"] = (
                    "The session was created but could NOT be configured from "
                    f"{cpacs_file_path}. It holds a minimal config that SU2 will "
                    "reject; call configure_from_cpacs with a valid CPACS path "
                    "before running the solver."
                )
            else:
                result["configured"] = True
                result["references"] = configured.get("references")
        else:
            result["configured"] = False
            result["warning"] = (
                "This session holds a MINIMAL config that SU2 will reject (no "
                "solver, markers or numerics). Pass cpacs_file_path to "
                "create_su2_session, or call configure_from_cpacs, before "
                "run_su2_solver."
            )
        return result
    except Exception as exc:  # pragma: no cover - defensive
        return _error("Failed to create SU2 session", details=str(exc))


def close_su2_session(
    session_id: str, delete_workdir: bool = False
) -> dict[str, object]:
    """Close a session and optionally delete its working directory."""
    try:
        success = SESSION_MANAGER.close_session(
            session_id, delete_workdir=delete_workdir
        )
        return {"success": success}
    except Exception as exc:  # pragma: no cover
        return _error("Failed to close session", details=str(exc))


def get_session_info(session_id: str) -> dict[str, object]:
    """Return session paths and last run metadata."""
    try:
        return SESSION_MANAGER.to_info(session_id)
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to read session info", details=str(exc))


__all__ = [
    "create_su2_session",
    "close_su2_session",
    "get_session_info",
    "SESSION_MANAGER",
]
