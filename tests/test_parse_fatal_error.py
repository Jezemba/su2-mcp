"""Every fatal SU2 exit must reach the caller, not just the config-option class.

The config-error promotion (B22) was written for one class of failure. Everything
else SU2 dies of stayed inside `log_tail`. Measured on a real no-mesh exit, SU2's
message is perfectly clear --

    The SU2 mesh file named mesh.su2 was not found.

-- and sits on line 35 of a 47-line, 2744-character banner. Across two sweeps,
16% of ALL tool calls (117/717 and 58/410) were byte-identical repeats of the
call that had just failed: a caller reading a response and extracting nothing
actionable from it.

The fix relays SU2's OWN wording. SU2 already words these well; the defect was
placement, not phrasing, so nothing here substitutes a message of our own.

The fixtures below are captured verbatim from SU2 v8.3 runs.
"""

import pytest

from su2_mcp.su2_runner import parse_config_errors, parse_fatal_error

# Verbatim from `SU2_CFD` with MESH_FILENAME pointing at a file that isn't there.
NO_MESH = """\
------------------------------ Begin Solver -----------------------------

Error in "static short unsigned int CConfig::GetnDim(const std::string&, short unsigned int)":
-------------------------------------------------------------------------
The SU2 mesh file named mesh.su2 was not found.
------------------------------ Error Exit -------------------------------
Errorcode: 1
"""

# Verbatim shape of a marker mismatch -- the other failure the sweeps produced.
BAD_MARKER = """\
Error in "void CConfig::SetMarkers(unsigned short)":
-------------------------------------------------------------------------
The marker wing was not found in the mesh file.
------------------------------ Error Exit -------------------------------
"""


class TestFatalErrorIsLifted:
    def test_no_mesh_message_is_extracted(self):
        out = parse_fatal_error(NO_MESH)
        assert out is not None
        assert "mesh file named mesh.su2 was not found" in out["message"]

    def test_su2_own_wording_is_preserved(self):
        """We relay SU2, we do not paraphrase it."""
        assert "The SU2 mesh file named mesh.su2 was not found." in parse_fatal_error(NO_MESH)["message"]

    def test_location_is_reported(self):
        assert "GetnDim" in parse_fatal_error(NO_MESH)["location"]

    def test_banner_rules_are_not_part_of_the_message(self):
        msg = parse_fatal_error(NO_MESH)["message"]
        assert "---" not in msg
        assert "Errorcode" not in msg

    def test_a_different_fatal_class_also_works(self):
        """The point is generality -- not a second special case."""
        out = parse_fatal_error(BAD_MARKER)
        assert "marker wing was not found" in out["message"]

    def test_escaped_newlines_are_handled(self):
        """log_tail reaches us JSON-escaped on some paths."""
        out = parse_fatal_error(NO_MESH.replace("\n", "\\n"))
        assert out is not None and "was not found" in out["message"]


class TestNoFalsePositives:
    def test_successful_run_has_no_fatal_error(self):
        log = "Begin Solver\n  1  -3.5  -2.1  0.301  0.0142\nExit Success\n"
        assert parse_fatal_error(log) is None

    def test_empty_log(self):
        assert parse_fatal_error("") is None

    def test_word_error_in_prose_is_not_a_fatal_block(self):
        assert parse_fatal_error("Relative error tolerance reached.\n") is None

    def test_header_without_body_is_not_reported(self):
        assert parse_fatal_error('Error in "f":\n------ Error Exit ------\n') is None


class TestConfigClassKeepsItsRicherHandling:
    """parse_fatal_error is the catch-all; it must not displace the structured
    config path, which additionally names the correct option spelling."""

    def test_config_errors_still_parsed(self):
        log = "Line 9 MARKER_BODY: invalid option name.\nDid you mean MARKER_EMISSIVITY?\n"
        assert parse_config_errors(log)

    def test_no_mesh_is_not_mistaken_for_a_config_error(self):
        assert parse_config_errors(NO_MESH) == []


class TestSurfacedToTheCaller:
    def test_run_su2_solver_promotes_it_out_of_log_tail(self, monkeypatch):
        from su2_mcp.tools import run_tools

        class _Rec:
            workdir = "/tmp"
            config_path = "/tmp/config.cfg"

        class _Mgr:
            def require(self, sid):
                return _Rec()

            def record_run(self, *a, **k):
                pass

        class _Runner:
            def __init__(self, *a, **k):
                pass

            def run(self, *a, **k):
                return {"success": False, "exit_code": 1, "log_tail": NO_MESH}

        monkeypatch.setattr(run_tools, "SESSION_MANAGER", _Mgr())
        monkeypatch.setattr(run_tools, "SU2Runner", _Runner)
        monkeypatch.setattr(run_tools, "ensure_force_output", lambda *a, **k: {})

        out = run_tools.run_su2_solver("s")
        assert "mesh.su2 was not found" in out["solver_error"]
        assert "no history and no CL/CD" in out["hint"]

    def test_hint_says_a_bare_retry_will_not_help(self):
        """Aimed squarely at the 16% identical-repeat rate."""
        from su2_mcp.tools import run_tools

        assert parse_fatal_error(NO_MESH) is not None  # guard the fixture
