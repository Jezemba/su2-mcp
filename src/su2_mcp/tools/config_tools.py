"""Configuration-related MCP tools."""

from __future__ import annotations

from typing import Any

from su2_mcp import config_utils
from su2_mcp.tools.session import SESSION_MANAGER, _error


def get_config_text(session_id: str) -> dict[str, object]:
    """Return raw config text for the session."""
    try:
        record = SESSION_MANAGER.require(session_id)
        return {"config_text": record.config_path.read_text()}
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to read config", details=str(exc))


def parse_config(session_id: str) -> dict[str, object]:
    """Parse the session configuration into key/value entries."""
    try:
        record = SESSION_MANAGER.require(session_id)
        entries = config_utils.parse_config_file(record.config_path)
        return {"entries": entries}
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to parse config", details=str(exc))


def update_config_entries(
    session_id: str,
    updates: dict[str, Any],
    create_if_missing: bool = True,
) -> dict[str, object]:
    """Update configuration entries for a session.

    Deprecated SU2 v5/v6 option names are automatically remapped to their
    SU2 v8.3 equivalents.  Any remapped keys are returned in the response
    so the caller can learn the correct names for future calls.
    """
    try:
        record = SESSION_MANAGER.require(session_id)
        corrected, warnings = config_utils.remap_deprecated_keys(updates)
        updated = config_utils.update_config_entries(
            record.config_path, corrected, create_if_missing=create_if_missing
        )
        result: dict[str, object] = {"updated_keys": updated}
        if warnings:
            result["deprecated_remapped"] = warnings
            result["note"] = (
                "Some option names were deprecated in SU2 v8.3 and have been "
                "automatically remapped. Please use the new names in future calls."
            )

        # Check for missing required options and warn the caller.
        current = config_utils.parse_config_file(record.config_path)
        current_keys = {k.upper() for k in current}
        _REQUIRED_EULER = {
            "SOLVER", "MACH_NUMBER", "AOA",
            "FREESTREAM_PRESSURE", "FREESTREAM_TEMPERATURE",
            "MARKER_EULER", "MARKER_FAR",
            "CONV_NUM_METHOD_FLOW", "TIME_DISCRE_FLOW",
            "NUM_METHOD_GRAD", "CFL_NUMBER", "ITER",
        }
        missing = _REQUIRED_EULER - current_keys
        if missing:
            result["missing_required"] = sorted(missing)
            result["missing_note"] = (
                f"SU2 will fail without these options: {', '.join(sorted(missing))}. "
                "Call update_config_entries again to set them."
            )

        return result
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to update config", details=str(exc))


def set_mesh(
    session_id: str,
    mesh_base64: str,
    mesh_file_name: str = "mesh.su2",
    update_config: bool = True,
) -> dict[str, object]:
    """Persist a mesh file for the session and update the config."""
    try:
        mesh_path = SESSION_MANAGER.update_mesh(session_id, mesh_base64, mesh_file_name)
        if update_config:
            config_utils.update_config_entries(
                SESSION_MANAGER.require(session_id).config_path,
                {"MESH_FILENAME": mesh_file_name},
            )
        return {"mesh_path": str(mesh_path)}
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to set mesh", details=str(exc))


def get_valid_config_options() -> dict[str, object]:
    """Return commonly-used valid SU2 v8.3 configuration options.

    This is a discovery tool: it lists the ~30 most common Euler/RANS options
    with their expected types and brief descriptions so that an agent can
    construct correct ``update_config_entries`` calls without relying on
    outdated option names from earlier SU2 versions.
    """
    options = [
        # --- Problem definition ---
        {"name": "SOLVER", "type": "string", "description": "Solver type: EULER, NAVIER_STOKES, RANS, INC_EULER, etc."},
        {"name": "MATH_PROBLEM", "type": "string", "description": "DIRECT or ADJOINT."},
        # --- Freestream ---
        {"name": "MACH_NUMBER", "type": "float", "description": "Freestream Mach number."},
        {"name": "AOA", "type": "float", "description": "Angle of attack (degrees)."},
        {"name": "SIDESLIP_ANGLE", "type": "float", "description": "Sideslip angle (degrees)."},
        {"name": "FREESTREAM_PRESSURE", "type": "float", "description": "Freestream pressure (Pa)."},
        {"name": "FREESTREAM_TEMPERATURE", "type": "float", "description": "Freestream temperature (K)."},
        {"name": "GAMMA_VALUE", "type": "float", "description": "Ratio of specific heats."},
        {"name": "GAS_CONSTANT", "type": "float", "description": "Specific gas constant (J/(kg*K))."},
        # --- Reference values ---
        {"name": "REF_ORIGIN_MOMENT_X", "type": "float", "description": "X-coordinate of moment reference origin."},
        {"name": "REF_ORIGIN_MOMENT_Y", "type": "float", "description": "Y-coordinate of moment reference origin."},
        {"name": "REF_ORIGIN_MOMENT_Z", "type": "float", "description": "Z-coordinate of moment reference origin."},
        {"name": "REF_LENGTH", "type": "float", "description": "Reference length for moment computation."},
        {"name": "REF_AREA", "type": "float", "description": "Reference area for force coefficients."},
        # --- Markers ---
        {"name": "MARKER_EULER", "type": "list[string]", "description": "Euler (slip-wall) boundary markers."},
        {"name": "MARKER_FAR", "type": "list[string]", "description": "Farfield boundary markers."},
        {"name": "MARKER_MONITORING", "type": "list[string]", "description": "Markers for force/moment monitoring."},
        {"name": "MARKER_PLOTTING", "type": "list[string]", "description": "Markers included in surface output."},
        # --- Numerical methods ---
        {"name": "NUM_METHOD_GRAD", "type": "string", "description": "Gradient method: GREEN_GAUSS or WEIGHTED_LEAST_SQUARES."},
        {"name": "CONV_NUM_METHOD_FLOW", "type": "string", "description": "Convective flux scheme: JST, LAX-FRIEDRICH, ROE, AUSM, HLLC, etc."},
        {"name": "TIME_DISCRE_FLOW", "type": "string", "description": "Time integration: EULER_IMPLICIT, EULER_EXPLICIT, RUNGE-KUTTA_EXPLICIT."},
        {"name": "CFL_NUMBER", "type": "float", "description": "CFL number for pseudo-time stepping."},
        {"name": "CFL_ADAPT", "type": "string", "description": "NO or YES (adaptive CFL)."},
        {"name": "MGLEVEL", "type": "int", "description": "Number of multigrid levels (0 = single grid)."},
        # --- Linear solver ---
        {"name": "LINEAR_SOLVER", "type": "string", "description": "Linear solver: FGMRES, BCGSTAB, SMOOTHER."},
        {"name": "LINEAR_SOLVER_PREC", "type": "string", "description": "Preconditioner: ILU, LU_SGS, LINELET, JACOBI."},
        {"name": "LINEAR_SOLVER_ERROR", "type": "float", "description": "Linear solver convergence tolerance."},
        {"name": "LINEAR_SOLVER_ITER", "type": "int", "description": "Max linear solver iterations per nonlinear step."},
        # --- Convergence ---
        {"name": "ITER", "type": "int", "description": "Maximum number of outer iterations (replaces deprecated EXT_ITER)."},
        {"name": "CONV_FIELD", "type": "string", "description": "Field for convergence monitoring, e.g. RMS_DENSITY (replaces CONV_CRITERIA)."},
        {"name": "CONV_RESIDUAL_MINVAL", "type": "float", "description": "Log10 residual threshold for convergence (replaces RESIDUAL_REDUCTION)."},
        {"name": "CONV_STARTITER", "type": "int", "description": "Iteration to begin convergence checking (replaces STARTCONV_ITER)."},
        # --- Mesh ---
        {"name": "MESH_FILENAME", "type": "string", "description": "Path to the mesh file."},
        {"name": "MESH_FORMAT", "type": "string", "description": "Mesh format: SU2, CGNS."},
        {"name": "TABULAR_FORMAT", "type": "string", "description": "Tabular output format: CSV, TECPLOT."},
        # --- Output ---
        {"name": "OUTPUT_FILES", "type": "list[string]", "description": "Output file types: RESTART, PARAVIEW, SURFACE_PARAVIEW, etc. (replaces OUTPUT_FORMAT)."},
        {"name": "SCREEN_OUTPUT", "type": "list[string]", "description": "Fields shown on screen each iteration."},
        {"name": "HISTORY_OUTPUT", "type": "list[string]", "description": "Fields written to history file."},
        {"name": "OUTPUT_WRT_FREQ", "type": "int", "description": "Iteration frequency for writing output files (replaces WRT_SOL_FREQ)."},
    ]
    deprecated = [
        {"old": old, "new": new}
        for old, new in config_utils.DEPRECATED_OPTIONS.items()
    ]
    return {
        "options": options,
        "deprecated_mappings": deprecated,
        "note": (
            "These are the most common Euler/RANS options for SU2 v8.3. "
            "Deprecated v5/v6 names are listed in deprecated_mappings; "
            "update_config_entries will auto-remap them but prefer using "
            "the new names directly."
        ),
    }


__all__ = [
    "get_config_text",
    "get_valid_config_options",
    "parse_config",
    "update_config_entries",
    "set_mesh",
]
