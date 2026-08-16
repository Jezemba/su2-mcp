"""Utilities for parsing and modifying SU2 configuration files."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from pathlib import Path

# --------------------------------------------------------------------------- #
# SU2 v5/v6 → v8.3 deprecated option name mapping
# --------------------------------------------------------------------------- #

DEPRECATED_OPTIONS: dict[str, str] = {
    "PHYSICAL_PROBLEM": "SOLVER",
    "CONV_CRITERIA": "CONV_FIELD",
    "RESIDUAL_REDUCTION": "CONV_RESIDUAL_MINVAL",
    "RESIDUAL_MINVAL": "CONV_RESIDUAL_MINVAL",
    "OUTPUT_FORMAT": "OUTPUT_FILES",
    "WRT_SOL_FREQ": "OUTPUT_WRT_FREQ",
    "WRT_CON_FREQ": "OUTPUT_WRT_FREQ",
    "SCREEN_WRT_FREQ": "OUTPUT_WRT_FREQ",
    "STARTCONV_ITER": "CONV_STARTITER",
    "EXT_ITER": "ITER",
    # Freestream Mach. MACH is the single most dangerous entry in this map:
    # SU2's own `did_you_mean` answers "MACH_MOTION" by string similarity, and
    # MACH_MOTION is a VALID option (moving-mesh Mach) that SU2 accepts without
    # complaint -- so following the suggestion turns a loud rejection into a
    # silently wrong cruise point. Observed live 2026-08-12: a caller wrote
    # MACH, was steered to MACH_MOTION, then guessed FREESTREAM_MACH, six
    # config rejections and no solve. Same trap as PHYSICAL_PROBLEM drawing
    # "MATH_PROBLEM"; remapping here promotes the correct option ahead of SU2's
    # spelling match.
    "MACH": "MACH_NUMBER",
    "FREESTREAM_MACH": "MACH_NUMBER",
    "MACH_INF": "MACH_NUMBER",
    "NUM_METHOD": "NUM_METHOD_GRAD",
    "AOA_DEG": "AOA",
    "ANGLE_OF_ATTACK": "AOA",
}


# Options SU2 has been OBSERVED to reject, beyond the deprecated names above.
# Kept separate from DEPRECATED_OPTIONS because these have no correct
# equivalent to remap to -- they are simply wrong, and forwarding one rejects
# the ENTIRE config, so a single bad key kills a solve whose every pinned value
# is correct (observed 2026-08-12: TURB_MODEL blocked a run four times).
#
# NOT validated against `get_valid_config_options`: that is a curated discovery
# list of ~39 common options, and seven CANONICAL settings are absent from it
# (KIND_TURB_MODEL, MGCYCLE, JST_SENSOR_COEFF, REYNOLDS_NUMBER,
# CFL_ADAPT_PARAM, REF_DIMENSIONALIZATION, RESTART_SOL). Using it as a validator
# would silently strip pinned numerics from every config -- worse than the bug.
# So this drops only what SU2 itself has rejected.
KNOWN_INVALID_OPTIONS: frozenset[str] = frozenset({
    "TURB_MODEL",        # correct name is KIND_TURB_MODEL
    "NUM_METHOD",        # correct name is NUM_METHOD_GRAD
    "PHYSICAL_PROBLEM",  # v6 name for SOLVER (also in DEPRECATED_OPTIONS)
    "MACH",              # correct name is MACH_NUMBER
    "FREESTREAM_MACH",
    "MACH_INF",
})


def drop_known_invalid(
    updates: MutableMapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Remove options SU2 is known to reject. Returns (kept, dropped_names).

    Run AFTER remap_deprecated_keys, so anything with a correct equivalent has
    already been renamed and only genuinely unusable keys remain.
    """
    kept, dropped = {}, []
    for key, value in updates.items():
        if str(key).upper() in KNOWN_INVALID_OPTIONS:
            dropped.append(str(key))
        else:
            kept[key] = value
    return kept, dropped


def remap_deprecated_keys(
    updates: MutableMapping[str, object],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Replace deprecated option names with SU2 8.3 equivalents.

    Returns a new dict with corrected keys and a list of warning dicts
    ``[{"old": ..., "new": ...}, ...]`` describing what was remapped.
    """
    remapped: dict[str, object] = {}
    warnings: list[dict[str, str]] = []
    for key, value in updates.items():
        new_key = DEPRECATED_OPTIONS.get(key)
        if new_key is not None:
            warnings.append({"old": key, "new": new_key})
            remapped[new_key] = value
        else:
            remapped[key] = value
    return remapped, warnings


def _infer_scalar(value: str) -> object:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_config_text(config_text: str) -> dict[str, object]:
    """Parse SU2 configuration text into a dictionary."""
    entries: dict[str, object] = {}
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", maxsplit=1)]
        if not key:
            continue
        if "," in value:
            parts = [segment.strip() for segment in value.split(",") if segment.strip()]
            entries[key] = [_infer_scalar(part) for part in parts]
        else:
            entries[key] = _infer_scalar(value)
    return entries


def parse_config_file(config_path: Path) -> dict[str, object]:
    """Parse a configuration file from disk."""
    return parse_config_text(config_path.read_text())


def _format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def update_config_entries(
    config_path: Path,
    updates: MutableMapping[str, object],
    create_if_missing: bool = True,
) -> list[str]:
    """Update configuration entries and persist them to disk."""
    entries = parse_config_file(config_path)
    updated_keys: list[str] = []

    for key, value in updates.items():
        if key in entries or create_if_missing:
            entries[key] = value
            updated_keys.append(key)

    serialized_lines = _serialize_entries(entries.items())
    config_path.write_text("\n".join(serialized_lines) + "\n")
    return updated_keys


def _serialize_entries(entries: Iterable[tuple[str, object]]) -> list[str]:
    return [f"{key}= {_format_value(value)}" for key, value in entries]


# --------------------------------------------------------------------------- #
# Force-output guarantee
# --------------------------------------------------------------------------- #

# Field group that must be in HISTORY_OUTPUT for history.csv to carry CL/CD.
# AERO_COEFF alone expands to CL, CD, CSF, CMx, CMy, CMz — it is sufficient.
# Do NOT also add LIFT/DRAG: they expand to CL and CD again, and SU2 then emits
# DUPLICATE columns (observed live: ..."CD","CD","CL","CL","CSF",...). Harmless
# to SU2 but confusing to any reader. If the author already listed LIFT/DRAG we
# leave them alone — this only controls what we ADD.
_FORCE_HISTORY_FIELDS = ("AERO_COEFF",)
# Keys naming the solid-wall marker, in preference order. MARKER_MONITORING is
# derived from whichever is present — SU2 computes forces ONLY on monitored
# markers, so without it there is no CL/CD no matter what HISTORY_OUTPUT says.
_WALL_MARKER_KEYS = ("MARKER_EULER", "MARKER_HEATFLUX", "MARKER_ISOTHERMAL", "MARKER_PLOTTING")


def _as_list(value: object) -> list[str]:
    """Normalise an SU2 list-valued option to bare tokens.

    ``parse_config_text`` splits on commas, so ``( ITER, RMS_RES, LIFT )``
    already arrives as a LIST whose first/last items still carry the
    parentheses (``'( ITER'`` … ``'LIFT )'``). Strip them per item, otherwise a
    membership test for ``AERO_COEFF`` misses ``'AERO_COEFF )'`` and a
    correctly configured file gets "fixed" every run.
    """
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = [str(value)]
    tokens: list[str] = []
    for item in items:
        cleaned = item.replace("(", " ").replace(")", " ").replace(",", " ")
        tokens.extend(part for part in cleaned.split() if part)
    return tokens


def ensure_force_output(config_path: Path) -> dict[str, object]:
    """Guarantee the config will actually WRITE force coefficients.

    An SU2 run can converge perfectly and still produce a history.csv with only
    residual columns::

        Time_Iter, Outer_Iter, Inner_Iter, "rms[Rho]", "rms[RhoU]", ...

    That happened in a real MAS-Aviary networked run (2026-08-02): the agent ran
    the solver AND read the history AND re-applied the mission parameters, but
    there were never any CL/CD to capture, so the mission silently flew on
    aviary's default drag polar. The aero stage "succeeded" while being
    physically incapable of coupling.

    Two independent things are required and this enforces both:

      1. ``MARKER_MONITORING`` — SU2 integrates forces ONLY over monitored
         markers. If it is absent, no amount of HISTORY_OUTPUT produces CL/CD.
         Derived from the solid-wall marker already in the config.
      2. ``HISTORY_OUTPUT`` must include LIFT / DRAG / AERO_COEFF.

    Non-destructive: existing values are preserved and only missing pieces are
    added, so an agent that configured things correctly is untouched. Returns a
    report of what was added (empty ``added`` == config was already fine).
    """
    entries = parse_config_file(config_path)
    updates: dict[str, object] = {}
    added: list[str] = []
    notes: list[str] = []

    # 1. MARKER_MONITORING — derive from the wall marker if absent/empty.
    monitoring = _as_list(entries.get("MARKER_MONITORING", ""))
    if not monitoring:
        wall: list[str] = []
        source = None
        for key in _WALL_MARKER_KEYS:
            if key in entries:
                candidate = _as_list(entries[key])
                if candidate:
                    wall, source = candidate, key
                    break
        if wall:
            updates["MARKER_MONITORING"] = f"( {', '.join(wall)} )"
            added.append("MARKER_MONITORING")
            notes.append(
                f"MARKER_MONITORING was missing — derived ( {', '.join(wall)} ) from {source}. "
                "SU2 integrates forces only over monitored markers, so without it "
                "history.csv carries no CL/CD."
            )
        else:
            notes.append(
                "MARKER_MONITORING is missing and no solid-wall marker "
                f"({'/'.join(_WALL_MARKER_KEYS)}) was found to derive it from — "
                "this run CANNOT produce force coefficients."
            )

    # 2. HISTORY_OUTPUT must carry the force fields.
    history = _as_list(entries.get("HISTORY_OUTPUT", ""))
    if not history:
        updates["HISTORY_OUTPUT"] = "( ITER, RMS_RES, LIFT, DRAG, AERO_COEFF )"
        added.append("HISTORY_OUTPUT")
        notes.append("HISTORY_OUTPUT was missing — set to include ITER, RMS_RES and the force fields.")
    else:
        upper = {h.upper() for h in history}
        missing = [f for f in _FORCE_HISTORY_FIELDS if f not in upper]
        if missing:
            merged = history + missing
            updates["HISTORY_OUTPUT"] = f"( {', '.join(merged)} )"
            added.append("HISTORY_OUTPUT")
            notes.append(
                f"HISTORY_OUTPUT was missing {', '.join(missing)} — appended so "
                "history.csv carries CL/CD."
            )

    # 3. Reference quantities. Writing forces is not enough — they must be
    #    NON-DIMENSIONALISED against the real aircraft. SU2 defaults REF_AREA to
    #    1.0 m^2, so a config without it yields coefficients scaled by the whole
    #    wing area. Live on 2026-08-03: a missing REF_AREA (true ~192 m^2) gave
    #    CL = 15.18 instead of ~0.08 -- a factor of ~190 -- which was injected
    #    into aviary, diverged its trim solver, and produced GTOW 310,918 kg /
    #    fuel 208,551 kg while still reporting itself "coupled".
    #
    #    Verified against the official SU2 v8.3.0 source:
    #      Common/src/CConfig.cpp:1488
    #        addDoubleOption("REF_AREA", RefArea, 1.0);          <- default 1.0
    #      Common/src/geometry/CPhysicalGeometry.cpp:4363
    #        if (config->GetRefArea() == 0.0)
    #          config->SetRefArea(TotalPositiveZArea);           <- 0 == AUTO
    #
    #    So REF_AREA=0 makes SU2 compute the positive-Z projected (planform)
    #    area from the actual mesh. That is BETTER than any value we could
    #    supply: it tracks the morphed geometry automatically, which is exactly
    #    what a coupled MDO wants, and it needs no knowledge we do not have.
    #    When REF_AREA is absent we therefore set 0 rather than guess a number
    #    or leave SU2 on its meaningless 1.0 default.
    ref_area_raw = entries.get("REF_AREA")
    ref_area_bad = False
    if ref_area_raw is None:
        updates["REF_AREA"] = 0
        added.append("REF_AREA")
        notes.append(
            "REF_AREA was missing — set to 0, which makes SU2 auto-compute the "
            "projected planform area from the mesh (CPhysicalGeometry.cpp:4363). "
            "Left unset, SU2 would use its 1.0 m^2 default and every coefficient "
            "would be off by roughly the wing area (observed: CL 15.18 vs ~0.08)."
        )
    else:
        try:
            ref_area_val = float(ref_area_raw)
        except (TypeError, ValueError):
            ref_area_val = None  # type: ignore[assignment]
        if ref_area_val is None:
            ref_area_bad = True
            notes.append(f"REF_AREA is not numeric ({ref_area_raw!r}) — coefficients may be wrong.")
        elif ref_area_val == 1.0:
            # Exactly SU2's default. Could be the default leaking through, or a
            # deliberate unit-chord 2D case — so warn, do NOT override an
            # explicit value.
            ref_area_bad = True
            notes.append(
                "REF_AREA is exactly 1.0, SU2's default. For an aircraft mesh this is "
                "almost certainly unintended and makes CL/CD wrong by roughly the wing "
                "area. Set REF_AREA=0 to have SU2 compute the planform area from the "
                "mesh, or give the real reference area."
            )

    if updates:
        update_config_entries(config_path, updates, create_if_missing=True)

    monitoring_ok = bool(
        _as_list(entries.get("MARKER_MONITORING", "")) or "MARKER_MONITORING" in updates
    )
    return {
        "force_output_ok": monitoring_ok,
        # Forces will be WRITTEN but are only meaningful as coefficients when the
        # reference quantities are real.
        "coefficients_meaningful": monitoring_ok and not ref_area_bad,
        "ref_area": updates.get("REF_AREA", ref_area_raw),
        "added": added,
        "notes": notes,
    }
