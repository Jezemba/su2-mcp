"""SU2's config complaints must reach the caller as actionable items.

Measured across two MAS-Aviary sweeps (2026-08-03/04): SU2 failed **42 of 46**
and **20 of 25** solve attempts, roughly two thirds of them on invalid config
options. That — not agent ordering, and not a skipped read_history_csv — is why
the aero coupling almost never happened: a solve that never starts leaves no
history and therefore no CL/CD to capture.

SU2 already reports exactly what is wrong AND the correct spelling. The problem
was that it lived inside the `log_tail` prose blob, so the caller could not act
on it. Every observed mistake is a near-miss (GREEN-GAUSS vs GREEN_GAUSS,
IMPLICIT vs EULER_IMPLICIT), so a structured error should let the caller fix it
in one step.

The log text below is verbatim from the failing runs.
"""

import pytest

from su2_mcp.su2_runner import parse_config_errors

# Verbatim from the rerun log (JSON-escaped newlines, as captured).
REAL_LOG = (
    '\\n\\nError in \\"void CConfig::SetConfig_Parsing(std::istream&)\\": '
    "\\n-------------------------------------------------------------------------"
    "\\nLine 9 MARKER_BODY: invalid option name. Check current SU2 options in "
    "config_template.cfg.\\nDid you mean MARKER_EMISSIVITY?"
    "\\nTIME_DISCRE_FLOW: invalid option value IMPLICIT."
    "\\nDid you mean, ADER_DG, CLASSICAL_RK4_EXPLICIT, EULER_EXPLICIT, "
    "EULER_IMPLICIT, RUNGE-KUTTA_EXPLICIT?"
    "\\n\\n------------------------------ Error Exit -----------------------------"
)


class TestRealSU2Output:
    def test_both_errors_found(self):
        errs = parse_config_errors(REAL_LOG)
        assert len(errs) == 2

    def test_invalid_option_name_with_line_and_suggestion(self):
        e = parse_config_errors(REAL_LOG)[0]
        assert e["option"] == "MARKER_BODY"
        assert e["problem"] == "invalid option name"
        assert e["line"] == 9
        assert e["did_you_mean"] == ["MARKER_EMISSIVITY"]

    def test_invalid_option_value_with_all_suggestions(self):
        e = parse_config_errors(REAL_LOG)[1]
        assert e["option"] == "TIME_DISCRE_FLOW"
        assert e["problem"] == "invalid option value"
        assert e["value"] == "IMPLICIT"
        # The fix the agent actually needed is in the list.
        assert "EULER_IMPLICIT" in e["did_you_mean"]
        assert len(e["did_you_mean"]) == 5


class TestOtherObservedMistakes:
    def test_hyphen_vs_underscore_near_miss(self):
        """GREEN-GAUSS vs GREEN_GAUSS — seen live."""
        log = "NUM_METHOD_GRAD: invalid option value GREEN-GAUSS.\nDid you mean, GREEN_GAUSS, WEIGHTED_LEAST_SQUARES?"
        e = parse_config_errors(log)[0]
        assert e["option"] == "NUM_METHOD_GRAD"
        assert e["value"] == "GREEN-GAUSS"
        assert "GREEN_GAUSS" in e["did_you_mean"]

    def test_real_newlines_also_parsed(self):
        """Logs arrive with real newlines or escaped ones depending on capture."""
        log = "Line 3 FOO: invalid option name.\nDid you mean BAR?"
        e = parse_config_errors(log)[0]
        assert e["option"] == "FOO" and e["did_you_mean"] == ["BAR"]

    def test_error_without_suggestion_still_reported(self):
        e = parse_config_errors("Line 5 WEIRD_OPT: invalid option name.")[0]
        assert e["option"] == "WEIRD_OPT"
        assert "did_you_mean" not in e


class TestNoFalsePositives:
    @pytest.mark.parametrize("log", [
        "",
        "All convergence criteria satisfied.\nExit Success",
        "Iter 100 rms[Rho] -6.01 CL 0.2059 CD 0.0027",
    ])
    def test_clean_logs_yield_nothing(self, log):
        assert parse_config_errors(log) == []

    def test_non_config_failure_yields_nothing(self):
        """A solve that died for another reason must not be misreported."""
        log = "MPI_ABORT was invoked on rank 0\nErrorcode: 1\nSegmentation fault"
        assert parse_config_errors(log) == []

    def test_none_and_empty_safe(self):
        assert parse_config_errors("") == []
