"""Avion B151 (2026-10-08): a rename that changes the meaning is refused, not applied.

MAS-Aviary v2b149: RESIDUAL_REDUCTION 6 became CONV_RESIDUAL_MINVAL= 6 and NUM_METHOD "Jacobian" became an
invalid NUM_METHOD_GRAD; SU2 died in 1.6 s and the stage produced no aero.
"""

import pytest

from su2_mcp import config_utils
from su2_mcp.tools.config_tools import update_config_entries  # noqa: F401  (import check)


@pytest.mark.parametrize("updates", [{"RESIDUAL_REDUCTION": "6"}, {"NUM_METHOD": "Jacobian"},
                                     {"CONV_RESIDUAL_MINVAL": "6"}, {"CONV_RESIDUAL_MINVAL": 3}])
def test_meaning_changing_updates_are_refused(updates):
    kept, rejected = config_utils.reject_meaning_changes(updates)
    assert kept == {} and len(rejected) == 1 and rejected[0]["reason"]


def test_the_v2b149_update_keeps_only_the_safe_keys():
    kept, rejected = config_utils.reject_meaning_changes(
        {"MAX_ITER": "15000", "RESIDUAL_REDUCTION": "6", "NUM_METHOD": "Jacobian", "INNER_ITER": "10"})
    assert sorted(r["key"] for r in rejected) == ["NUM_METHOD", "RESIDUAL_REDUCTION"]
    assert kept == {"MAX_ITER": "15000", "INNER_ITER": "10"}


def test_a_real_gradient_method_under_num_method_is_still_renamed():
    kept, rejected = config_utils.reject_meaning_changes({"NUM_METHOD": "GREEN_GAUSS"})
    assert kept == {"NUM_METHOD_GRAD": "GREEN_GAUSS"} and rejected == []


def test_a_negative_threshold_and_the_same_meaning_remap_pass():
    kept, rejected = config_utils.reject_meaning_changes({"CONV_RESIDUAL_MINVAL": "-6", "RESIDUAL_MINVAL": "-8"})
    assert rejected == [] and kept == {"CONV_RESIDUAL_MINVAL": "-6", "RESIDUAL_MINVAL": "-8"}
    remapped, _ = config_utils.remap_deprecated_keys(kept)
    assert remapped["CONV_RESIDUAL_MINVAL"] == "-8"


def test_the_trim_uses_the_measured_lift_slope():
    from su2_mcp.tools import config_tools
    import inspect

    assert '"DCL_DALPHA": 0.075' in inspect.getsource(config_tools)


def test_the_trim_can_converge_and_ends_before_the_watchdog():
    """Avion B152: CONV_CAUCHY_EPS (default 1e-10) is SU2's CL tolerance in fixed-CL mode."""
    from su2_mcp.tools import config_tools
    import inspect

    src = inspect.getsource(config_tools)
    assert '"CONV_CAUCHY_EPS": 0.01' in src and '"ITER_DCL_DALPHA": 0,' in src
    assert config_tools._TRIM_MIN_ITER * 8.4 < 3600     # worst case at the measured ~8.4 s/iteration
