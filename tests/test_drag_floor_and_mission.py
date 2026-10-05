"""Induced-drag floor on the reported CD (Avion B138) and the mission flight state (Avion B132)."""

import math
import textwrap

import pytest

from su2_mcp import config_utils, drag_model
from su2_mcp.tools import config_tools

CPACS = textwrap.dedent("""\
    <?xml version="1.0"?>
    <cpacs><vehicles><aircraft><model><reference>
      <area>122.4</area>
      <length>4.193</length>
    </reference></model></aircraft></vehicles></cpacs>
""")


# ---- drag model -------------------------------------------------------------------------------

def test_raymer_span_efficiency_values():
    assert drag_model.oswald_efficiency(9.37) == pytest.approx(0.773, abs=2e-3)
    assert drag_model.oswald_efficiency(11.5) == pytest.approx(0.718, abs=2e-3)


def test_a_cd_above_the_floor_is_reported_unchanged():
    a = drag_model.assess_drag(cl=0.2, cd=0.004, aspect_ratio=9.37)
    assert a["source"] == "su2" and a["cd_reported"] == 0.004
    assert a["cd_induced_floor"] == pytest.approx(0.04 / (math.pi * 9.37 * a["oswald_efficiency"]))


@pytest.mark.parametrize("cd", [-0.0059, -0.0021, 0.00069])
def test_a_cd_below_the_floor_reports_the_induced_drag(cd):
    """The measured cases: three morphed designs (negative) and the D150 baseline (+0.00069)."""
    a = drag_model.assess_drag(cl=0.2396, cd=cd, aspect_ratio=9.37)
    assert a["source"] == "induced_floor"
    assert a["cd_reported"] == pytest.approx(a["cd_induced_floor"]) and a["cd_reported"] > 0
    assert a["cd_su2"] == cd                                          # raw value kept


def test_a_caller_efficiency_is_used():
    a = drag_model.assess_drag(cl=0.2, cd=-0.001, aspect_ratio=10.0, oswald=0.85)
    assert a["oswald_efficiency"] == 0.85 and a["oswald_source"] == "caller"


def test_without_aspect_ratio_the_cd_is_unchecked():
    a = drag_model.assess_drag(cl=0.2, cd=-0.003, aspect_ratio=None)
    assert a["source"] == "su2_unchecked" and a["cd_reported"] == -0.003


def test_isa_at_fl330():
    """US Standard Atmosphere at 10,058 m: 26.20 kPa, 222.77 K. (MAS-Aviary's canonical pins
    26,436 Pa, 0.9 % above standard; its temperature 222.8 K agrees.)"""
    p, t = drag_model.isa(33000)
    assert p == pytest.approx(26200, rel=1e-3) and t == pytest.approx(222.77, abs=0.02)


def test_isa_above_the_tropopause():
    p, t = drag_model.isa(38000)
    assert t == pytest.approx(216.65, abs=0.01) and p == pytest.approx(20646, rel=5e-3)


# ---- configure_from_cpacs ---------------------------------------------------------------------

@pytest.fixture
def session(tmp_path, monkeypatch):
    cpacs = tmp_path / "d150.xml"
    cpacs.write_text(CPACS)
    cfg = tmp_path / "config.cfg"
    cfg.write_text("% empty\n")

    class _Rec:
        config_path = cfg
        workdir = tmp_path
        aspect_ratio = None
        oswald_efficiency = None

    rec = _Rec()

    class _Mgr:
        def require(self, sid):
            return rec

    monkeypatch.setattr(config_tools, "SESSION_MANAGER", _Mgr())
    return {"sid": "s", "cpacs": str(cpacs), "cfg": cfg, "rec": rec}


def test_mission_flight_state_is_written_and_wins_over_overrides(session):
    out = config_tools.configure_from_cpacs(
        session["sid"], session["cpacs"],
        overrides={"MACH_NUMBER": 0.5, "FREESTREAM_PRESSURE": 101325.0},
        cruise_mach=0.78, cruise_altitude_ft=36000)
    e = config_utils.parse_config_file(session["cfg"])
    assert float(e["MACH_NUMBER"]) == 0.78
    p, t = drag_model.isa(36000)
    assert float(e["FREESTREAM_PRESSURE"]) == pytest.approx(p, abs=0.1)
    assert float(e["FREESTREAM_TEMPERATURE"]) == pytest.approx(t, abs=0.01)
    assert out["flight_state"]["altitude_ft"] == 36000.0


def test_aspect_ratio_is_stored_on_the_session(session):
    out = config_tools.configure_from_cpacs(session["sid"], session["cpacs"],
                                            aspect_ratio=9.37, oswald_efficiency=0.8)
    assert session["rec"].aspect_ratio == 9.37 and session["rec"].oswald_efficiency == 0.8
    assert out["drag_floor"]["aspect_ratio"] == 9.37


def test_without_mission_args_the_flight_state_is_untouched(session):
    out = config_tools.configure_from_cpacs(session["sid"], session["cpacs"],
                                            overrides={"MACH_NUMBER": 0.7})
    assert float(config_utils.parse_config_file(session["cfg"])["MACH_NUMBER"]) == 0.7
    assert "note" in out["flight_state"] and "note" in out["drag_floor"]


# ---- the real tools end to end (no SU2 binary needed) -------------------------------------------

def test_create_session_passes_everything_through(tmp_path):
    from su2_mcp.tools import session as session_tools

    cpacs = tmp_path / "d150.xml"
    cpacs.write_text(CPACS)
    out = session_tools.create_su2_session(cpacs_file_path=str(cpacs), aspect_ratio=9.37,
                                           cruise_mach=0.78, cruise_altitude_ft=33000, target_cl=0.55)
    assert out["configured"] is True
    rec = session_tools.SESSION_MANAGER.require(out["session_id"])
    assert rec.aspect_ratio == 9.37
    e = config_utils.parse_config_file(rec.config_path)
    assert float(e["MACH_NUMBER"]) == 0.78 and e["FIXED_CL_MODE"] == "YES"
    assert out["flight_state"]["FREESTREAM_PRESSURE"] == pytest.approx(26200, rel=1e-3)


def test_run_su2_solver_reports_the_assessed_drag(tmp_path, monkeypatch):
    from su2_mcp.tools import run_tools
    from su2_mcp.tools import session as session_tools

    cpacs = tmp_path / "d150.xml"
    cpacs.write_text(CPACS)
    sid = session_tools.create_su2_session(cpacs_file_path=str(cpacs), aspect_ratio=9.37)["session_id"]
    rec = session_tools.SESSION_MANAGER.require(sid)
    (rec.workdir / "history.csv").write_text(
        '"Inner_Iter","rms[Rho]","CL","CD"\n0,-1.0,0.30,0.2\n65,-6.0,0.2396,-0.0059\n')

    class _Runner:
        def __init__(self, workdir):
            pass

        def run(self, solver, config_path, max_runtime_seconds, capture_log_lines):
            return {"success": True, "solver": solver, "config_used": str(config_path),
                    "exit_code": 0, "runtime_seconds": 1.0, "log_tail": ""}

    monkeypatch.setattr(run_tools, "SU2Runner", _Runner)
    out = run_tools.run_su2_solver(sid)
    a = out["drag_assessment"]
    assert out["final_coefficients"]["CD"] == -0.0059
    assert a["source"] == "induced_floor" and a["cd_reported"] > 0
