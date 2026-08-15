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


class TestMissingRequiredOption:
    """SU2 8.3 reports a missing SOLVER by its DEPRECATED name, which SU2 8.3
    itself then rejects as invalid — an agent following the message literally
    loops. Observed live 2026-08-04."""

    REAL = (
        'Error in \\"void CConfig::SetPostprocessing(SU2_COMPONENT, short unsigned int, '
        'short unsigned int)\\": '
        "\\n-------------------------------------------------------------------------"
        "\\nPHYSICAL_PROBLEM must be set in the configuration file"
        "\\n------------------------------ Error Exit ------------------------------"
    )

    def test_missing_option_detected(self):
        e = parse_config_errors(self.REAL)
        assert len(e) == 1
        assert e[0]["option"] == "PHYSICAL_PROBLEM"
        assert e[0]["problem"] == "required option missing"

    def test_translated_to_the_name_su2_accepts(self):
        e = parse_config_errors(self.REAL)[0]
        assert e["did_you_mean"] == ["SOLVER"]
        assert "deprecated" in e["note"]

    def test_non_deprecated_missing_option_has_no_suggestion(self):
        e = parse_config_errors("MESH_FILENAME must be set in the configuration file")[0]
        assert e["option"] == "MESH_FILENAME"
        assert "did_you_mean" not in e


class TestSU2SuggestionIsOverriddenWhenWrong:
    """SU2's "Did you mean" is a STRING-SIMILARITY guess. For deprecated options
    it is semantically wrong, and relaying it verbatim sends the caller nowhere
    useful. Observed live 2026-08-04: PHYSICAL_PROBLEM (the v6 name for SOLVER)
    drew "Did you mean MATH_PROBLEM?"."""

    REAL = (
        "Line 1 PHYSICAL_PROBLEM: invalid option name. Check current SU2 options in "
        "config_template.cfg.\nDid you mean MATH_PROBLEM?"
    )

    def test_correct_option_is_offered_first(self):
        e = parse_config_errors(self.REAL)[0]
        assert e["did_you_mean"][0] == "SOLVER"

    def test_su2s_wrong_guess_is_kept_but_demoted(self):
        e = parse_config_errors(self.REAL)[0]
        assert "MATH_PROBLEM" in e["did_you_mean"]
        assert e["did_you_mean"].index("SOLVER") < e["did_you_mean"].index("MATH_PROBLEM")

    def test_note_explains_the_override(self):
        e = parse_config_errors(self.REAL)[0]
        assert "deprecated name for SOLVER" in e["note"]
        assert "spelling match" in e["note"]

    def test_mach_is_promoted_over_su2s_dangerous_suggestion(self):
        """This test used to assert the opposite, on the assumption that SU2
        suggests MACH_NUMBER for MACH and should be left alone. Live output
        2026-08-12 disproved it:

            {"option": "MACH", "did_you_mean": ["MACH_MOTION"]}

        MACH_MOTION is the MOVING-MESH Mach number and is a VALID option, so SU2
        accepts it silently -- following the suggestion turns a loud rejection
        into a quietly wrong cruise point. Exactly the PHYSICAL_PROBLEM ->
        MATH_PROBLEM trap, which is why MACH now sits in DEPRECATED_OPTIONS and
        the correct option is promoted ahead of the spelling match.
        """
        e = parse_config_errors(
            "Line 4 MACH: invalid option name.\nDid you mean MACH_MOTION?"
        )[0]
        assert e["did_you_mean"][0] == "MACH_NUMBER"
        assert "note" in e

    def test_a_genuinely_unknown_option_keeps_su2s_suggestion(self):
        """Promotion must only apply where we KNOW the right answer."""
        e = parse_config_errors(
            "Line 4 WOBBLE_FACTOR: invalid option name.\nDid you mean WOBBLE?"
        )[0]
        assert e["did_you_mean"] == ["WOBBLE"]
        assert "note" not in e


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
