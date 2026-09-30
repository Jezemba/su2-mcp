"""B111: the solve must be trimmed to the lift the aircraft needs, not flown at a fixed angle.

geoauth_all8_7960: every solve ran at AOA 2 deg, producing CL 0.05-0.22 while cruise for this
aircraft is near 0.55. At that off-design lift the inviscid pressure drag is ~1e-4 and the mesh
errs by ~1e-2, so 5 of 16 links reported a NEGATIVE, converged, stable CD -- which the drag
coupling then clamped to its 0.5 floor and flew.
"""
from pathlib import Path

import pytest

from su2_mcp.cpacs_adapter import _fixed_cl_block


def test_no_target_writes_nothing():
    assert _fixed_cl_block({"aoa_deg": 2.0}) == ""
    assert _fixed_cl_block({"aoa_deg": 2.0, "target_cl": None}) == ""


def test_a_target_turns_on_fixed_cl_mode():
    block = _fixed_cl_block({"target_cl": 0.55})
    assert "FIXED_CL_MODE= YES" in block
    assert "TARGET_CL= 0.55" in block
    assert "DCL_DALPHA=" in block
    # SU2 8.3 rejects UPDATE_ALPHA, and one bad key rejects the whole config.
    assert "UPDATE_AOA_ITER_LIMIT=" in block and "UPDATE_ALPHA=" not in block


def _config(tmp_path, monkeypatch, **kwargs):
    from su2_mcp.tools import config_tools

    cpacs = tmp_path / "d.xml"
    cpacs.write_text("<cpacs><vehicles/></cpacs>")
    (tmp_path / "config.cfg").write_text("SOLVER= EULER\n")   # create_su2_session writes one first

    class _Record:
        config_path = tmp_path / "config.cfg"
        workdir = tmp_path

    class _Mgr:
        def require(self, sid):
            return _Record()

    monkeypatch.setattr(config_tools, "SESSION_MANAGER", _Mgr())
    out = config_tools.configure_from_cpacs("s", str(cpacs), **kwargs)
    return out, (tmp_path / "config.cfg").read_text()


def test_the_config_carries_the_target(tmp_path, monkeypatch):
    out, text = _config(tmp_path, monkeypatch, target_cl=0.55)
    assert "FIXED_CL_MODE= YES" in text
    assert "TARGET_CL= 0.55" in text
    assert out["trim"]["target_cl"] == 0.55
    assert "UPDATE_ALPHA=" not in text          # SU2 8.3 rejects it -- and one bad key kills the config
    assert "UPDATE_AOA_ITER_LIMIT= 100" in text


def test_trimming_raises_the_iteration_budget(tmp_path, monkeypatch):
    """A fixed-AOA solve converges in 60-100 iterations; the trim needs room after each update."""
    out, text = _config(tmp_path, monkeypatch, target_cl=0.55)
    assert "ITER= 500" in text


def test_a_caller_pinned_iteration_count_is_kept(tmp_path, monkeypatch):
    out, text = _config(tmp_path, monkeypatch, target_cl=0.55, overrides={"ITER": 900})
    assert "ITER= 900" in text


def test_every_option_written_is_valid_for_this_su2(tmp_path, monkeypatch):
    """Probed against SU2_CFD 8.3.0: an invalid option name rejects the entire config."""
    import shutil
    import subprocess

    su2 = shutil.which("SU2_CFD") or "/home/jezemba/.conda/envs/su2env/bin/SU2_CFD"
    if not Path(su2).exists():
        pytest.skip("SU2_CFD not installed here")
    _out, text = _config(tmp_path, monkeypatch, target_cl=0.55)
    probe = tmp_path / "probe.cfg"
    probe.write_text(text)
    result = subprocess.run([su2, str(probe)], capture_output=True, text=True, timeout=120)
    assert "invalid option name" not in (result.stdout + result.stderr)


def test_without_a_target_the_config_is_unchanged_and_says_so(tmp_path, monkeypatch):
    out, text = _config(tmp_path, monkeypatch)
    assert "FIXED_CL_MODE" not in text
    assert out["trim"]["target_cl"] is None
    assert "not necessarily a cruise condition" in out["trim"]["note"]


# ---- the solved geometry must be inspectable (B115) ----------------------------------------------

def test_the_surface_is_written_for_inspection(tmp_path, monkeypatch):
    """surface_flow.vtu is what you open in ParaView to SEE what was solved.

    Without it, nothing in the pipeline ever showed the aircraft, and a mesh carrying one wing and
    one tailplane (B115) was solved in every study on this machine without anyone seeing it.
    OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW) is SU2's own default.
    """
    _out, text = _config(tmp_path, monkeypatch)
    assert "SURFACE_PARAVIEW" in text
    assert "SURFACE_FILENAME= surface_flow" in text
