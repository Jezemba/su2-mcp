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
    assert "UPDATE_ALPHA=" in block and "DCL_DALPHA=" in block


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


def test_without_a_target_the_config_is_unchanged_and_says_so(tmp_path, monkeypatch):
    out, text = _config(tmp_path, monkeypatch)
    assert "FIXED_CL_MODE" not in text
    assert out["trim"]["target_cl"] is None
    assert "not necessarily a cruise condition" in out["trim"]["note"]
