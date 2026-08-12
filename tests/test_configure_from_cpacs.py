"""SU2 configs come FROM the CPACS geometry, not from an agent's transcription.

Across two MAS-Aviary sweeps SU2 rejected 80-91% of agent-written configs —
PHYSICAL_PROBLEM (the v6 name for SOLVER), GREEN-GAUSS (for GREEN_GAUSS),
IMPLICIT (for EULER_IMPLICIT), MARKER_BODY, MACH — and a solve that never starts
leaves no history, so the aero coupling could not happen. The identical pipeline
driven programmatically from the same canonical values worked every time. The
difference was transcription, not physics.

cpacs_adapter.py already read REF_AREA/REF_LENGTH from CPACS and emitted a
correct template, but it was never exposed as a tool — dead code from the
agents' point of view. This exposes that path.

The split under test:
  * CPACS owns the geometry references (they must track the morphed design).
  * The caller owns flight state + numerics — the adapter's own defaults are
    SEA LEVEL (101325 Pa / 288.15 K), wrong for a cruise case.
"""

import textwrap

import pytest

from su2_mcp import config_utils
from su2_mcp.tools import config_tools

CPACS = textwrap.dedent("""\
    <?xml version="1.0"?>
    <cpacs><vehicles><aircraft><model><reference>
      <area>239.4</area>
      <length>5.219</length>
    </reference></model></aircraft></vehicles></cpacs>
""")

# The F25 cruise numerics/flight-state pinned by the experiment's canonical
# baseline — ISA at 33 kft, JST, multigrid.
CANONICAL = {
    "FREESTREAM_PRESSURE": 26436.0,
    "FREESTREAM_TEMPERATURE": 222.8,
    "CONV_NUM_METHOD_FLOW": "JST",
    "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
    "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
    "CFL_NUMBER": 20,
    "MGLEVEL": 3,
    "ITER": 120,
}


@pytest.fixture
def session(tmp_path, monkeypatch):
    cpacs = tmp_path / "f25.xml"
    cpacs.write_text(CPACS)
    cfg = tmp_path / "config.cfg"
    cfg.write_text("% empty\n")

    class _Rec:
        config_path = cfg
        workdir = tmp_path

    class _Mgr:
        def require(self, sid):
            return _Rec()

    monkeypatch.setattr(config_tools, "SESSION_MANAGER", _Mgr())
    return {"sid": "s", "cpacs": str(cpacs), "cfg": cfg}


def _entries(session):
    return config_utils.parse_config_file(session["cfg"])


class TestReferencesComeFromCPACS:
    def test_ref_area_and_length_read_from_geometry(self, session):
        out = config_tools.configure_from_cpacs(session["sid"], session["cpacs"])
        assert out["references"]["REF_AREA"] == pytest.approx(239.4)
        assert out["references"]["REF_LENGTH"] == pytest.approx(5.219)
        e = _entries(session)
        assert float(e["REF_AREA"]) == pytest.approx(239.4)
        assert float(e["REF_LENGTH"]) == pytest.approx(5.219)

    def test_morphed_geometry_changes_the_reference(self, tmp_path, session):
        """The reference must TRACK the design, which is the whole point."""
        morphed = tmp_path / "morphed.xml"
        morphed.write_text(CPACS.replace("<area>239.4</area>", "<area>122.4</area>"))
        out = config_tools.configure_from_cpacs(session["sid"], str(morphed))
        assert out["references"]["REF_AREA"] == pytest.approx(122.4)

    def test_missing_cpacs_is_reported(self, session):
        out = config_tools.configure_from_cpacs(session["sid"], "/nope/missing.xml")
        assert out["error"]["type"] == "not_found"


class TestComputedGeometryBeatsDeclaredCPACS:
    """Morphing does NOT update CPACS's <reference><area>. Measured on a real
    morphed export: the file still declared area 122.4 while get_wing_summary
    computed 65.98 for that same geometry. Trusting the declaration would feed
    SU2 a stale reference and reproduce the wrong-CL bug (B12)."""

    def test_explicit_reference_wins(self, session):
        out = config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], ref_area=65.985, ref_length=4.193
        )
        assert out["references"]["REF_AREA"] == pytest.approx(65.985)
        assert out["references"]["ref_area_source"] == "computed_geometry"
        assert float(_entries(session)["REF_AREA"]) == pytest.approx(65.985)

    def test_declared_fallback_is_flagged_as_possibly_stale(self, session):
        out = config_tools.configure_from_cpacs(session["sid"], session["cpacs"])
        assert out["references"]["ref_area_source"] == "cpacs_declared"
        assert any("BASELINE" in w for w in out["reference_warnings"])

    def test_no_warning_when_both_supplied(self, session):
        out = config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], ref_area=65.985, ref_length=4.193
        )
        assert "reference_warnings" not in out


class TestForceOutputGuaranteedByConstruction:
    def test_marker_monitoring_present(self, session):
        """Without it SU2 integrates no forces and history has no CL/CD."""
        config_tools.configure_from_cpacs(session["sid"], session["cpacs"])
        assert "aircraft" in str(_entries(session)["MARKER_MONITORING"])

    def test_history_output_carries_force_fields(self, session):
        config_tools.configure_from_cpacs(session["sid"], session["cpacs"])
        assert "AERO_COEFF" in str(_entries(session)["HISTORY_OUTPUT"]).upper()


class TestCallerOwnsFlightStateAndNumerics:
    def test_overrides_win(self, session):
        config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], overrides=CANONICAL
        )
        e = _entries(session)
        assert float(e["FREESTREAM_PRESSURE"]) == pytest.approx(26436.0)
        assert float(e["FREESTREAM_TEMPERATURE"]) == pytest.approx(222.8)
        assert str(e["CONV_NUM_METHOD_FLOW"]) == "JST"
        assert int(e["ITER"]) == 120

    def test_overrides_do_not_clobber_cpacs_references(self, session):
        config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], overrides=CANONICAL
        )
        e = _entries(session)
        assert float(e["REF_AREA"]) == pytest.approx(239.4)

    def test_deprecated_override_names_are_remapped(self, session):
        """A caller passing the v6 spelling still gets a valid config."""
        out = config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], overrides={"PHYSICAL_PROBLEM": "EULER"}
        )
        e = _entries(session)
        assert str(e["SOLVER"]) == "EULER"
        assert "PHYSICAL_PROBLEM" not in e
        assert out["deprecated_remapped"]


class TestNoneOfTheObservedMistakesSurvive:
    @pytest.mark.parametrize("bad", ["PHYSICAL_PROBLEM", "MARKER_BODY", "MACH"])
    def test_agent_invented_options_are_absent(self, session, bad):
        config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], overrides=CANONICAL
        )
        assert bad not in _entries(session)

    def test_valid_solver_and_gradient_values(self, session):
        config_tools.configure_from_cpacs(
            session["sid"], session["cpacs"], overrides=CANONICAL
        )
        e = _entries(session)
        assert str(e["SOLVER"]) == "EULER"
        assert str(e["NUM_METHOD_GRAD"]) == "WEIGHTED_LEAST_SQUARES"
        assert str(e["TIME_DISCRE_FLOW"]) == "EULER_IMPLICIT"
