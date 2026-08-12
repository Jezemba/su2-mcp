"""A new SU2 session must be usable without a follow-up call.

A session created without a CPACS path holds a minimal config that SU2 rejects
outright (no solver, markers or numerics). Recovering requires a SECOND call to
configure_from_cpacs -- and callers do not reliably make it.

Measured live (sweep 2026-08-12, run 5/16, orchestrated_staged_pipeline):

    configure_from_cpacs advertised to the agent : True
    responses naming configure_from_cpacs        : 5
    configure_from_cpacs calls                   : 0
    create_su2_session / run_su2_solver          : 6 / 6
    next action after each hint                  : final_answer

The run ended with no mission output at all. The generalisation, from comparing
this with the tigl UID fix: an error supplying a VALUE to substitute into the
call being made gets acted on (recovery went from a mean 8.8 wrong guesses to
1), while an error asking for a DIFFERENT TOOL does not. So the fix cannot be
advisory -- the session has to arrive usable.
"""

import textwrap

import pytest

from su2_mcp import config_utils
from su2_mcp.tools import session as session_tools

CPACS = textwrap.dedent("""\
    <?xml version="1.0"?>
    <cpacs><vehicles><aircraft><model><reference>
      <area>239.4</area><length>5.219</length>
    </reference></model></aircraft></vehicles></cpacs>
""")


@pytest.fixture
def cpacs(tmp_path):
    p = tmp_path / "f25.xml"
    p.write_text(CPACS)
    return str(p)


def _entries(result):
    from pathlib import Path
    return config_utils.parse_config_file(Path(str(result["config_path"])))


class TestConfiguredOnCreation:
    def test_session_is_solvable_without_a_second_call(self, cpacs):
        out = session_tools.create_su2_session(cpacs_file_path=cpacs)
        assert out["configured"] is True
        e = _entries(out)
        for required in ("SOLVER", "MARKER_EULER", "MARKER_FAR", "MARKER_MONITORING"):
            assert required in e, f"{required} missing -- SU2 would reject this"

    def test_references_come_from_the_geometry(self, cpacs):
        out = session_tools.create_su2_session(cpacs_file_path=cpacs)
        assert out["references"]["REF_AREA"] == pytest.approx(239.4)

    def test_computed_references_win(self, cpacs):
        """The data plane injects get_wing_summary values; they must be used."""
        out = session_tools.create_su2_session(
            cpacs_file_path=cpacs, ref_area=58.505, ref_length=3.282
        )
        assert out["references"]["REF_AREA"] == pytest.approx(58.505)

    def test_overrides_reach_the_config(self, cpacs):
        out = session_tools.create_su2_session(
            cpacs_file_path=cpacs, overrides={"ITER": 120, "CFL_NUMBER": 20}
        )
        assert int(_entries(out)["ITER"]) == 120


class TestUnconfiguredSessionSaysSo:
    def test_no_cpacs_path_warns_it_cannot_solve(self):
        out = session_tools.create_su2_session()
        assert out["configured"] is False
        assert "SU2 will reject" in out["warning"]

    def test_warning_names_both_ways_to_fix_it(self):
        w = session_tools.create_su2_session()["warning"]
        assert "cpacs_file_path" in w and "configure_from_cpacs" in w

    def test_session_is_still_created(self):
        """An unusable config is not a reason to fail the call."""
        out = session_tools.create_su2_session()
        assert out["session_id"]


class TestBadCpacsDoesNotDiscardTheSession:
    def test_session_survives_a_bad_path(self):
        out = session_tools.create_su2_session(cpacs_file_path="/nope/missing.xml")
        assert out["session_id"], "a good session must not be lost to a config failure"
        assert out["configured"] is False
        assert "configure_error" in out

    def test_it_says_what_actually_failed(self):
        out = session_tools.create_su2_session(cpacs_file_path="/nope/missing.xml")
        assert "could NOT be configured" in out["warning"]
