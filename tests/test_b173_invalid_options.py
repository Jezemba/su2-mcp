"""Avion B173: invalid option names must not stick in a session's config (aipexws8 v2b172, 2026-10-10).

The aero agent put this tool's own arguments into ``overrides``:
    overrides={"cruise_mach": 0.78, "cruise_altitude_ft": 33000, "aspect_ratio": 10.5075, "target_cl": None}
They were written into config.cfg; SU2 rejected the whole config ("CRUISE_MACH: invalid option name", ...).
The hint said to use did_you_mean spellings or configure_from_cpacs, but configure_from_cpacs merged into
the existing file and update_config_entries could not delete a key, so every solve failed the same way.
"""

import textwrap

import pytest

from su2_mcp import config_utils
from su2_mcp.tools import config_tools, run_tools
from su2_mcp.tools import session as session_tools

CPACS = textwrap.dedent("""\
    <?xml version="1.0"?>
    <cpacs><vehicles><aircraft><model><reference>
      <area>122.4</area>
      <length>4.193</length>
    </reference></model></aircraft></vehicles></cpacs>
""")
AGENT_OVERRIDES = {"cruise_mach": 0.78, "cruise_altitude_ft": 33000, "aspect_ratio": 10.5075, "target_cl": None}
UCX = "[1760123456.1] [host:123] UCX  ERROR cuDeviceGetCount() failed: no CUDA-capable device is detected\n"


def _keys(sid):
    rec = session_tools.SESSION_MANAGER.require(sid)
    return {k.lower() for k in config_utils.parse_config_file(rec.config_path)}, rec


def _cpacs(tmp_path):
    p = tmp_path / "d150.xml"
    p.write_text(CPACS)
    return str(p)


def test_tool_arguments_in_overrides_are_used_as_arguments_not_written(tmp_path):
    out = session_tools.create_su2_session(cpacs_file_path=_cpacs(tmp_path), overrides=dict(AGENT_OVERRIDES))
    keys, rec = _keys(out["session_id"])
    assert not keys & {"cruise_mach", "cruise_altitude_ft", "aspect_ratio", "target_cl"}
    assert out["arguments_taken_from_overrides"]["cruise_mach"] == 0.78
    assert out["arguments_taken_from_overrides"]["target_cl"].startswith("ignored")
    assert rec.aspect_ratio == pytest.approx(10.5075)
    e = config_utils.parse_config_file(rec.config_path)
    assert float(e["MACH_NUMBER"]) == 0.78
    assert float(e["FREESTREAM_PRESSURE"]) == pytest.approx(26200, rel=1e-3)


def test_an_explicit_argument_wins_over_the_same_name_in_overrides(tmp_path):
    out = session_tools.create_su2_session(cpacs_file_path=_cpacs(tmp_path), cruise_mach=0.8,
                                           overrides={"cruise_mach": 0.6})
    rec = session_tools.SESSION_MANAGER.require(out["session_id"])
    assert float(config_utils.parse_config_file(rec.config_path)["MACH_NUMBER"]) == 0.8


def _failing_runner(monkeypatch, log):
    class _Runner:
        def __init__(self, workdir):
            pass

        def run(self, solver, config_path, max_runtime_seconds, capture_log_lines):
            return {"success": False, "exit_code": 1, "runtime_seconds": 1.0, "log_tail": log}

    monkeypatch.setattr(run_tools, "SU2Runner", _Runner)


def test_a_poisoned_session_recovers_with_configure_from_cpacs(tmp_path, monkeypatch):
    """The v2b172 state: bad lines already in the file (written by the old server)."""
    cpacs = _cpacs(tmp_path)
    sid = session_tools.create_su2_session(cpacs_file_path=cpacs)["session_id"]
    rec = session_tools.SESSION_MANAGER.require(sid)
    with open(rec.config_path, "a") as f:
        f.write("cruise_mach= 0.78\ncruise_altitude_ft= 33000\nWOBBLE_FACTOR= 2\n")
    _failing_runner(monkeypatch, UCX * 66 + "Line 40 WOBBLE_FACTOR: invalid option name.\n")
    run = run_tools.run_su2_solver(sid)
    assert "REMOVE it" in run["hint"] and "did_you_mean spellings" not in run["hint"]
    assert "UCX" not in run["log_tail"] and run["log_noise_removed"].startswith("66 lines")
    fixed = config_tools.configure_from_cpacs(sid, cpacs)
    assert fixed["removed_invalid_options"] == ["WOBBLE_FACTOR", "cruise_altitude_ft", "cruise_mach"]
    keys, _ = _keys(sid)
    assert not keys & {"wobble_factor", "cruise_mach", "cruise_altitude_ft"}
    assert "mach_number" in keys


def test_update_config_entries_deletes_with_null_and_rejects_tool_arguments(tmp_path):
    sid = session_tools.create_su2_session(cpacs_file_path=_cpacs(tmp_path))["session_id"]
    rec = session_tools.SESSION_MANAGER.require(sid)
    with open(rec.config_path, "a") as f:
        f.write("BAD_NAME= 1\n")
    out = config_tools.update_config_entries(sid, {"BAD_NAME": None, "aspect_ratio": 10.5, "CFL_NUMBER": 5})
    assert out["removed_keys"] == ["BAD_NAME"]
    assert out["rejected_tool_arguments"] == ["aspect_ratio"]
    keys, _ = _keys(sid)
    assert "bad_name" not in keys and "aspect_ratio" not in keys and "cfl_number" in keys
