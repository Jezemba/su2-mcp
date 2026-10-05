"""Which drag to report from an inviscid (Euler) solve, and the flight state for a cruise point.

Why (Avion B116 / B138): SU2 integrates drag from surface pressure, and on an affordable mesh the
pressure forces that nearly cancel leave an error as large as the drag itself. At ~2.3M cells the
D150 whole aircraft converges to CD +0.00069 (forces flat for 40 iterations, so not an iterative
error) while its induced drag alone is about +0.0023; three morphed designs converged NEGATIVE
(-0.0059, -0.0024, -0.0021). Lift is resolved; absolute inviscid drag is not.

The rule here: an inviscid flow cannot have less drag than the induced drag of the lift it makes,
CD_i = CL^2 / (pi * AR * e). If SU2's CD is at or above that floor it is reported unchanged. If it
is below, the floor is reported instead and the response says so; the raw SU2 value is always kept.
Skin friction is NOT part of either number (an Euler solve has none); the caller adds it.

Span efficiency e: Raymer, Aircraft Design: A Conceptual Approach (6th ed.), eq. 12.48,
e = 1.78 (1 - 0.045 AR^0.68) - 0.64, stated for wings with leading-edge sweep up to about 30 deg.
It gives 0.77 at AR 9.4 and 0.72 at AR 11.5. A caller may pass its own value instead.
"""

from __future__ import annotations

import math
from typing import Any

ISA_T0_K = 288.15
ISA_P0_PA = 101325.0
ISA_LAPSE_K_PER_M = 0.0065
ISA_TROPOPAUSE_M = 11000.0
FT_TO_M = 0.3048


def oswald_efficiency(aspect_ratio: float) -> float:
    """Raymer eq. 12.48 span efficiency, held to a plausible band."""
    e = 1.78 * (1.0 - 0.045 * aspect_ratio**0.68) - 0.64
    return min(max(e, 0.5), 0.95)


def induced_drag(cl: float, aspect_ratio: float, oswald: float) -> float:
    return cl * cl / (math.pi * aspect_ratio * oswald)


def assess_drag(cl: float | None, cd: float | None, aspect_ratio: float | None,
                oswald: float | None = None) -> dict[str, Any]:
    """The drag to report for an inviscid solve, with the reason.

    ``source`` is ``"su2"`` when SU2's own CD is kept, ``"induced_floor"`` when it fell below the
    induced drag of its lift, and ``"su2_unchecked"`` when no aspect ratio was given, so no floor
    could be computed.
    """
    out: dict[str, Any] = {"cd_su2": cd, "cl": cl}
    if cl is None or cd is None:
        out.update({"cd_reported": cd, "source": "su2_unchecked",
                    "note": "No CL/CD to assess."})
        return out
    if not aspect_ratio or aspect_ratio <= 0:
        out.update({"cd_reported": cd, "source": "su2_unchecked",
                    "note": "No aspect_ratio was given to the session, so the induced-drag floor "
                            "was not applied. Pass aspect_ratio to create_su2_session or "
                            "configure_from_cpacs."})
        return out
    e = float(oswald) if oswald else oswald_efficiency(aspect_ratio)
    floor = induced_drag(cl, aspect_ratio, e)
    keep = cd >= floor
    out.update({
        "aspect_ratio": aspect_ratio,
        "oswald_efficiency": e,
        "oswald_source": "caller" if oswald else "Raymer eq. 12.48",
        "cd_induced_floor": floor,
        "cd_reported": cd if keep else floor,
        "source": "su2" if keep else "induced_floor",
        "note": ("SU2's CD is at or above the induced drag of its lift, so it is reported unchanged."
                 if keep else
                 "SU2's CD is below the induced drag of its lift, which an inviscid flow cannot be: "
                 "the surface-pressure integration error is larger than the drag on this mesh. "
                 "The induced drag CL^2/(pi AR e) is reported instead; cd_su2 keeps the raw value."),
    })
    return out


def isa(altitude_ft: float) -> tuple[float, float]:
    """ISA static (pressure Pa, temperature K) at a geopotential altitude in feet (to 20 km)."""
    h = float(altitude_ft) * FT_TO_M
    if h <= ISA_TROPOPAUSE_M:
        t = ISA_T0_K - ISA_LAPSE_K_PER_M * h
        p = ISA_P0_PA * (t / ISA_T0_K) ** 5.255877
    else:
        t11 = ISA_T0_K - ISA_LAPSE_K_PER_M * ISA_TROPOPAUSE_M
        p11 = ISA_P0_PA * (t11 / ISA_T0_K) ** 5.255877
        t = t11
        p = p11 * math.exp(-9.80665 * (h - ISA_TROPOPAUSE_M) / (287.05287 * t11))
    return p, t
