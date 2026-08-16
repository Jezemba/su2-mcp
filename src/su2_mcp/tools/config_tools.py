"""Configuration-related MCP tools."""

from __future__ import annotations

import binascii
from pathlib import Path
from typing import Any

from su2_mcp import config_utils
from su2_mcp.tools.session import SESSION_MANAGER, _error


def configure_from_cpacs(
    session_id: str,
    cpacs_file_path: str,
    overrides: dict[str, Any] | None = None,
    mesh_file_name: str = "mesh.su2",
    ref_area: float | None = None,
    ref_length: float | None = None,
) -> dict[str, object]:
    """Build this session's SU2 config FROM the CPACS file, not by hand.

    Reference quantities are the aircraft's, so they must come from the aircraft
    definition rather than being retyped. This reads ``REF_AREA`` and
    ``REF_LENGTH`` straight out of the CPACS geometry and writes a complete,
    valid Euler config around them -- including ``MARKER_MONITORING`` and force
    fields in ``HISTORY_OUTPUT``, without which SU2 solves happily and writes no
    CL/CD at all.

    Why this exists: agents were composing the whole config by hand from a dict
    embedded in their prompt, and drifting. Across two MAS-Aviary sweeps SU2
    rejected 80-91% of configs -- ``PHYSICAL_PROBLEM`` (the v6 name for SOLVER),
    ``GREEN-GAUSS`` (for GREEN_GAUSS), ``IMPLICIT`` (for EULER_IMPLICIT),
    ``MARKER_BODY``, ``MACH`` -- and a solve that never starts leaves no history,
    so the aero coupling could not happen. The identical pipeline driven
    programmatically, from the same canonical values, worked every time. The
    difference was transcription, not physics.

    ``overrides`` carries the caller's pinned flight state and numerics (e.g.
    cruise freestream and solver settings from an experiment's canonical
    baseline) and is applied LAST, so it wins. The split is deliberate:
      * CPACS owns the geometry references -- they must track the morphed design.
      * The caller owns flight state and numerics -- the adapter's built-in
        defaults are SEA-LEVEL (101325 Pa / 288.15 K), which is wrong for a
        cruise case, so a caller solving at altitude must override them.

    Returns the resulting entries plus which references came from CPACS, so the
    caller can verify the geometry actually reached the solver.
    """
    try:
        record = SESSION_MANAGER.require(session_id)

        path = Path(cpacs_file_path)
        if not path.is_file():
            return _error(
                f"CPACS file not found: {cpacs_file_path}",
                error_type="not_found",
            )

        from su2_mcp.cpacs_adapter import read_from_cpacs

        derived = read_from_cpacs(path.read_text(encoding="utf-8"))

        # CPACS's <reference><area> is a DECLARED value and morphing does NOT
        # update it. Measured on a real morphed export: the file still said
        # area 122.4 / length 4.193 while TiGL's get_wing_summary computed
        # 65.98 for the same geometry. Trusting the declared value would feed
        # SU2 a stale reference and reproduce the wrong-CL class of bug.
        #
        # So an explicitly supplied reference (from get_wing_summary, i.e. the
        # geometry as actually built) always wins; the declared value is only a
        # fallback, and is flagged as possibly stale when used.
        ref_notes: list[str] = []
        if ref_area is not None:
            derived["ref_area_m2"] = float(ref_area)
            ref_source_area = "computed_geometry"
        else:
            ref_source_area = "cpacs_declared"
            ref_notes.append(
                "REF_AREA came from the CPACS <reference><area> element, which morphing "
                "does NOT update -- it may describe the BASELINE wing, not this design. "
                "Pass ref_area from get_wing_summary for a design-tracking value."
            )
        if ref_length is not None:
            derived["ref_length_m"] = float(ref_length)
            ref_source_length = "computed_geometry"
        else:
            ref_source_length = "cpacs_declared"
            ref_notes.append(
                "REF_LENGTH came from the CPACS declaration; pass ref_length "
                "(the MAC from get_wing_summary) for a design-tracking value."
            )

        # Base config: valid SU2 8.3 Euler setup with force output guaranteed.
        entries: dict[str, Any] = {
            "SOLVER": "EULER",
            "MATH_PROBLEM": "DIRECT",
            "MESH_FILENAME": mesh_file_name,
            "MESH_FORMAT": "SU2",
            "REF_DIMENSIONALIZATION": "DIMENSIONAL",
            # From the aircraft definition -- the whole point of this tool.
            "REF_AREA": derived["ref_area_m2"],
            "REF_LENGTH": derived["ref_length_m"],
            "MACH_NUMBER": derived["mach"],
            "AOA": derived["aoa_deg"],
            # Force output. Without MARKER_MONITORING SU2 integrates no forces,
            # so history.csv carries residuals only and there is no CL/CD.
            "MARKER_MONITORING": "( aircraft )",
            "MARKER_PLOTTING": "( aircraft )",
            "MARKER_EULER": "( aircraft )",
            "MARKER_FAR": "( farfield )",
            "HISTORY_OUTPUT": "( ITER, RMS_RES, AERO_COEFF )",
            "CONV_FILENAME": "history",
            "OUTPUT_FILES": "( RESTART, PARAVIEW )",
        }

        # Caller's pinned flight state / numerics win.
        dropped: list[str] = []
        if overrides:
            corrected, remapped = config_utils.remap_deprecated_keys(overrides)
            # A single unusable key rejects the ENTIRE config, so a solve dies
            # even when every pinned value is right (observed: TURB_MODEL, four
            # times in one run).
            corrected, dropped = config_utils.drop_known_invalid(corrected)
            entries.update(corrected)
        else:
            remapped = []

        config_utils.update_config_entries(
            record.config_path, entries, create_if_missing=True
        )

        result: dict[str, object] = {
            "config_path": str(record.config_path),
            "references": {
                "REF_AREA": derived["ref_area_m2"],
                "REF_LENGTH": derived["ref_length_m"],
                "ref_area_source": ref_source_area,
                "ref_length_source": ref_source_length,
            },
            "keys_written": sorted(entries),
            "overrides_applied": sorted(overrides) if overrides else [],
        }
        if ref_notes:
            result["reference_warnings"] = ref_notes
        if remapped:
            result["deprecated_remapped"] = remapped
        if dropped:
            result["dropped_invalid"] = dropped
            result["dropped_note"] = (
                f"Dropped {', '.join(dropped)} -- SU2 rejects these names and one "
                "of them makes it refuse the whole config. The canonical settings "
                "are unaffected."
            )
        return result
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to configure from CPACS", details=str(exc))


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
        corrected, dropped_invalid = config_utils.drop_known_invalid(corrected)
        updated = config_utils.update_config_entries(
            record.config_path, corrected, create_if_missing=create_if_missing
        )
        result: dict[str, object] = {"updated_keys": updated}
        if dropped_invalid:
            result["dropped_invalid"] = dropped_invalid
            result["dropped_note"] = (
                f"Dropped {', '.join(dropped_invalid)} -- SU2 rejects these names, "
                "and one bad key makes it refuse the entire config."
            )
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
            # The diagnosis here was always right; the REMEDY sent the caller
            # back into the loop that caused the problem. "Call
            # update_config_entries again to set them" invites building a valid
            # SU2 config key-by-key from memory -- which is precisely how 80-91%
            # of configs came to be rejected (B22), and how the run below ended
            # up with no markers at all:
            #
            #   solver_error: "The configuration file doesn't have any
            #                  definition for marker farfield"
            #   force_output: MARKER_MONITORING missing, no wall marker to
            #                 derive it from -- CANNOT produce force coefficients
            #
            # configure_from_cpacs writes every one of these from the aircraft
            # definition in one call. Naming the tool that solves the problem
            # beats naming the tool that reports it.
            markers = {m for m in missing if m.startswith("MARKER_")}
            rebuild = markers or len(missing) >= 4
            remedy = (
                "Prefer configure_from_cpacs(session_id, cpacs_file_path): it derives "
                "these from the aircraft definition -- including the markers, without "
                "which SU2 integrates no forces and there is no CL/CD -- instead of "
                "setting them one at a time from memory."
                if rebuild
                else "Call update_config_entries again to set them."
            )
            result["missing_note"] = (
                f"SU2 will fail without these options: {', '.join(sorted(missing))}. "
                + remedy
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
    except ValueError as exc:
        # Two DIFFERENT failures land here and must not be conflated:
        #
        #   binascii.Error  -- the string is not base64 at all
        #   ValueError      -- it decoded fine but is not a mesh (decode_mesh),
        #                      which already explains itself precisely
        #
        # Wrapping both in "is not decodable base64" produced the self-
        # contradictory "mesh_base64 is not decodable base64 (mesh_base64
        # decoded to 6 bytes that are not an SU2 mesh...)" -- an error message
        # that misdescribes its own cause, which is the defect class this whole
        # pass exists to remove. decode_mesh's message passes through untouched.
        if not isinstance(exc, binascii.Error):
            return _error(str(exc), error_type="invalid_payload")
        # Base64's own words -- "Incorrect padding", "Invalid base64-encoded
        # string" -- describe the SYMPTOM of passing a placeholder or a stray
        # token as mesh DATA, and name neither the argument at fault nor
        # anything to do next. Callers answered that by reissuing the identical
        # call (measured: 25 occurrences across 13 runs).
        return _error(
            f"mesh_base64 is not valid base64 ({exc}). It must carry the mesh "
            "FILE CONTENTS, base64-encoded -- not a file name, path, or stand-in "
            f"token (received {len(mesh_base64)} characters). Obtain the contents "
            "from the meshing step that produced this mesh, then pass them here.",
            error_type="invalid_payload",
        )
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
