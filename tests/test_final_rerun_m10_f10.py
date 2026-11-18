"""Final re-run fixes (Avion 2026-11-18): M10 convergence/trim reporting, F10 protected settings, M11 trim report.

No SU2 binary, network or live server: the solver is a fake runner that writes a history file.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from su2_mcp import config_utils
from su2_mcp.convergence import assess, solver_exit_message
from su2_mcp.tools import config_tools, results_tools, run_tools
from su2_mcp.tools import session as session_tools

CPACS = textwrap.dedent("""\
    <?xml version="1.0"?>
    <cpacs><vehicles><aircraft><model><reference>
      <area>122.4</area><length>4.193</length>
    </reference></model></aircraft></vehicles></cpacs>
""")

HEADER = '"Time_Iter","Outer_Iter","Inner_Iter",    "rms[Rho]"    ,       "CD"       ,       "CL"       \n'


@pytest.fixture
def cpacs(tmp_path):
    p = tmp_path / "d150.xml"
    p.write_text(CPACS)
    return str(p)


def _entries(path) -> dict:
    return config_utils.parse_config_file(Path(str(path)))


def _history(workdir: Path, rows: list[tuple[int, float, float, float]]) -> None:
    text = HEADER + "".join(f"0, 0, {it}, {res}, {cd}, {cl}\n" for it, res, cd, cl in rows)
    (workdir / "history.csv").write_text(text)


def _fake_runner(monkeypatch, rows, log):
    class _Runner:
        def __init__(self, workdir: Path) -> None:
            self.workdir = workdir

        def run(self, solver, config_path, max_runtime_seconds, capture_log_lines):
            _history(self.workdir, rows)
            return {"success": True, "solver": solver, "config_used": str(config_path), "exit_code": 0,
                    "runtime_seconds": 1.0, "log_tail": "tail", "residual_history": [],
                    "solver_exit_message": solver_exit_message(log)}

    monkeypatch.setattr(run_tools, "SU2Runner", _Runner)


# ---------------------------------------------------------------- M10: convergence and trim


def test_exit_message_is_read_from_the_log():
    assert solver_exit_message("...\nAll convergence criteria satisfied.\n") == "All convergence criteria satisfied."
    assert solver_exit_message("Maximum number of iterations reached (ITER = 500 ).") == \
        "Maximum number of iterations reached (ITER = 500)."
    assert solver_exit_message("nothing here") is None


def test_s_gr_12_unconverged_solve_is_not_a_result(cpacs, monkeypatch):
    """l20_final S_GR 12: 499/500 iterations, rms[Rho] -4.23 against -6, CL 0.989 against a ~0.56 target."""
    s = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.5587, ref_area=122.83,
                                         overrides={"ITER": 500, "CONV_FIELD": "RMS_DENSITY",
                                                    "CONV_RESIDUAL_MINVAL": -6})
    _fake_runner(monkeypatch, [(498, -4.3, 0.18, 0.98), (499, -4.23, 0.182, 0.989)],
                 "Maximum number of iterations reached (ITER = 500 ).")
    r = run_tools.run_su2_solver(s["session_id"])
    assert r["success"] is False and r["solver_exit_ok"] is True
    assert r["converged"] is False and r["trimmed"] is False
    assert "final_coefficients" not in r                     # nothing downstream can take it for a result
    assert r["last_iteration_coefficients"]["CL"] == pytest.approx(0.989)
    conv = r["convergence"]
    assert conv["final_residual"] == pytest.approx(-4.23) and conv["target_residual"] == -6
    assert conv["iterations"] == 500 and conv["max_iterations"] == 500
    assert "NOT converged" in r["not_a_result"] and "NOT trimmed" in r["not_a_result"]
    assert "0.989" in r["not_a_result"] and "0.5587" in r["not_a_result"]
    # read_history_csv and get_session_info carry the same verdict
    h = results_tools.read_history_csv(s["session_id"], "history.csv")
    assert h["convergence"]["converged"] is False
    assert session_tools.get_session_info(s["session_id"])["last_convergence"]["converged"] is False


def test_converged_and_trimmed_solve_reports_coefficients(cpacs, monkeypatch):
    s = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.56,
                                         overrides={"ITER": 500, "CONV_FIELD": "RMS_DENSITY",
                                                    "CONV_RESIDUAL_MINVAL": -6})
    _fake_runner(monkeypatch, [(252, -6.02, 0.0162, 0.5594)], "All convergence criteria satisfied.")
    r = run_tools.run_su2_solver(s["session_id"])
    assert r["success"] is True and r["converged"] is True and r["trimmed"] is True
    assert r["final_coefficients"]["CL"] == pytest.approx(0.5594)
    assert "Converged" in r["convergence"]["summary"] and "Trimmed" in r["convergence"]["summary"]


def test_converged_but_off_target_is_not_trimmed(cpacs, monkeypatch):
    s = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.56,
                                         overrides={"CONV_FIELD": "RMS_DENSITY", "CONV_RESIDUAL_MINVAL": -6})
    _fake_runner(monkeypatch, [(300, -6.1, 0.03, 0.60)], "All convergence criteria satisfied.")
    r = run_tools.run_su2_solver(s["session_id"])
    assert r["converged"] is True and r["trimmed"] is False and r["success"] is False
    assert r["convergence"]["trim"]["relative_error"] == pytest.approx(0.60 / 0.56 - 1)


def test_without_a_log_the_history_decides():
    cfg = {"CONV_FIELD": "RMS_DENSITY", "CONV_RESIDUAL_MINVAL": -6, "ITER": 500}
    rows = [{"Inner_Iter": "499", "rms[Rho]": "-4.23", "CL": "0.5", "CD": "0.02"}]
    out = assess(cfg, rows, None)
    assert out["converged"] is False and out["trimmed"] is None      # fixed-AOA: no lift target
    out = assess(cfg, [{"Inner_Iter": "200", "rms[Rho]": "-6.2", "CL": "0.5"}], None)
    assert out["converged"] is True


def test_unknown_convergence_is_said_not_guessed():
    out = assess({}, [{"Inner_Iter": "10", "CL": "0.5"}], None)
    assert out["converged"] is None and "could not be checked" in out["summary"]
    assert "CONV_RESIDUAL_MINVAL" in out["basis"]


def test_a_failed_run_clears_the_previous_verdict(cpacs, monkeypatch):
    s = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.56,
                                         overrides={"CONV_FIELD": "RMS_DENSITY", "CONV_RESIDUAL_MINVAL": -6})
    _fake_runner(monkeypatch, [(252, -6.02, 0.0162, 0.5594)], "All convergence criteria satisfied.")
    run_tools.run_su2_solver(s["session_id"])

    class _Fail:
        def __init__(self, workdir):
            pass

        def run(self, *a):
            return {"success": False, "solver": "SU2_CFD", "config_used": "", "exit_code": 1,
                    "runtime_seconds": 0.1, "log_tail": "boom", "residual_history": None}

    monkeypatch.setattr(run_tools, "SU2Runner", _Fail)
    run_tools.run_su2_solver(s["session_id"])
    assert session_tools.get_session_info(s["session_id"])["last_convergence"] is None


# ---------------------------------------------------------------- F10: protected settings


def test_o_gr_17_freestream_override_does_not_replace_the_flight_state(cpacs):
    """O_GR 17 passed FREESTREAM 24,800 Pa / 223.15 K in overrides; cruise_altitude_ft sets them."""
    out = session_tools.create_su2_session(
        cpacs_file_path=cpacs, cruise_mach=0.78, cruise_altitude_ft=33000, target_cl=0.56,
        overrides={"FREESTREAM_PRESSURE": 24800, "FREESTREAM_TEMPERATURE": 223.15, "MACH_NUMBER": 0.78,
                   "CFL_NUMBER": 20})
    cfg = _entries(out["config_path"])
    assert cfg["FREESTREAM_PRESSURE"] == pytest.approx(26200.8, abs=0.5)
    assert cfg["FREESTREAM_TEMPERATURE"] == pytest.approx(222.77, abs=0.01)
    refused = {d["key"]: d for d in out["overrides_not_applied"]}
    assert set(refused) == {"FREESTREAM_PRESSURE", "FREESTREAM_TEMPERATURE"}   # equal MACH is not a conflict
    assert refused["FREESTREAM_PRESSURE"]["asked"] == 24800
    assert "cruise_altitude_ft" in refused["FREESTREAM_PRESSURE"]["reason"]
    assert "CFL_NUMBER" in out["overrides_applied"] and "FREESTREAM_PRESSURE" not in out["overrides_applied"]


def test_overrides_cannot_undo_the_trim_or_the_measured_area(cpacs):
    out = session_tools.create_su2_session(
        cpacs_file_path=cpacs, target_cl=0.56, ref_area=105.0,
        overrides={"FIXED_CL_MODE": "NO", "DCL_DALPHA": 0.2, "ITER": 100, "REF_AREA": 122.4})
    cfg = _entries(out["config_path"])
    assert cfg["FIXED_CL_MODE"] == "YES" and cfg["DCL_DALPHA"] == 0.075
    assert cfg["ITER"] >= config_tools._TRIM_MIN_ITER and cfg["REF_AREA"] == pytest.approx(105.0)
    assert {d["key"] for d in out["overrides_not_applied"]} == {"FIXED_CL_MODE", "DCL_DALPHA", "ITER", "REF_AREA"}


def test_target_cl_in_overrides_is_handled_explicitly(cpacs):
    out = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.6,
                                           overrides={"TARGET_CL": 0.56})
    assert _entries(out["config_path"])["TARGET_CL"] == pytest.approx(0.6)
    note = out["arguments_taken_from_overrides"]["target_cl"]
    assert note.startswith("ignored") and "0.56" in note and "0.6" in note
    # alone, it is taken as the argument (B173) and the trim is built from it
    out = session_tools.create_su2_session(cpacs_file_path=cpacs, overrides={"TARGET_CL": 0.56})
    assert _entries(out["config_path"])["FIXED_CL_MODE"] == "YES"
    assert out["trim"]["target_cl"] == pytest.approx(0.56)


def test_fixed_cl_mode_alone_is_refused(cpacs):
    out = session_tools.create_su2_session(cpacs_file_path=cpacs, overrides={"FIXED_CL_MODE": "YES"})
    assert "FIXED_CL_MODE" not in _entries(out["config_path"])
    assert out["overrides_not_applied"][0]["key"] == "FIXED_CL_MODE"


def test_update_config_entries_is_protected_the_same_way(cpacs):
    out = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.56, cruise_altitude_ft=33000,
                                           ref_area=105.0)
    sid = out["session_id"]
    r = config_tools.update_config_entries(sid, {"TARGET_CL": None, "FIXED_CL_MODE": "NO",
                                                 "FREESTREAM_PRESSURE": 24800, "REF_AREA": 50,
                                                 "ITER": 50, "CFL_NUMBER": 10})
    # TARGET_CL is a tool-argument name (B173), the rest are refused here
    assert {d["key"] for d in r["refused_keys"]} == {"FIXED_CL_MODE", "FREESTREAM_PRESSURE", "REF_AREA", "ITER"}
    cfg = _entries(out["config_path"])
    assert cfg["FIXED_CL_MODE"] == "YES" and cfg["REF_AREA"] == pytest.approx(105.0)
    assert cfg["FREESTREAM_PRESSURE"] == pytest.approx(26200.8, abs=0.5) and cfg["CFL_NUMBER"] == 10
    assert "change the argument" in r["refused_note"]


def test_reconfigure_with_new_arguments_changes_owned_values(cpacs):
    out = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.56, ref_area=105.0)
    sid = out["session_id"]
    again = config_tools.configure_from_cpacs(sid, cpacs, target_cl=0.62, ref_area=110.0)
    cfg = _entries(again["config_path"])
    assert cfg["TARGET_CL"] == pytest.approx(0.62) and cfg["REF_AREA"] == pytest.approx(110.0)
    assert "overrides_not_applied" not in again


# ---------------------------------------------------------------- M11: the trim report says what the config holds


def test_reconfigure_without_target_reports_the_kept_trim(cpacs):
    out = session_tools.create_su2_session(cpacs_file_path=cpacs, target_cl=0.56)
    again = config_tools.configure_from_cpacs(out["session_id"], cpacs)
    cfg = _entries(again["config_path"])
    assert cfg["FIXED_CL_MODE"] == "YES" and cfg["TARGET_CL"] == pytest.approx(0.56)  # was deleted (M11)
    assert again["trim"]["target_cl"] == pytest.approx(0.56)
    assert "earlier call" in again["trim"]["note"]


def test_untrimmed_session_still_says_fixed_aoa(cpacs):
    out = session_tools.create_su2_session(cpacs_file_path=cpacs)
    assert out["trim"]["target_cl"] is None and "configured AOA" in out["trim"]["note"]
