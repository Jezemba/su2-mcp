"""Avion B267: a mesh passed as mesh_file_name froze the server.

l2c seq_graph, 2026-11-10: mesh_file_name arrived as the 155,514,712-character mesh. It was written into the
config as MESH_FILENAME, SU2 echoed it into its log, and parse_config_errors' lazy (\\S+?) patterns -- quadratic
in a line's length -- ran on that line with the server's only thread for hours.
"""
import base64
import os
import time
from pathlib import Path

import pytest

from su2_mcp import config_utils
from su2_mcp.session_manager import check_mesh_file_name
from su2_mcp.su2_runner import clip_log_lines, parse_config_errors, parse_fatal_error
from su2_mcp.tools.session import create_su2_session

HUGE = base64.b64encode(os.urandom(300_000)).decode()          # 400k characters, no spaces, no colon


@pytest.mark.parametrize("name", ["mesh.su2", "wing_mesh-2.su2", "a.cgns"])
def test_plain_names_pass(name):
    assert check_mesh_file_name(name) == name


@pytest.mark.parametrize("name", [HUGE, "", "../mesh.su2", "/tmp/mesh.su2", "mesh su2", "x" * 256, ".."])
def test_anything_else_is_refused_without_echoing_it(name):
    with pytest.raises(ValueError) as exc:
        check_mesh_file_name(name)
    assert len(str(exc.value)) < 600


def test_create_session_with_the_mesh_as_its_name_is_refused_briefly():
    out = create_su2_session(mesh_file_name=HUGE, initial_mesh=base64.b64encode(b"NDIME= 3\n").decode())
    assert out["error"]["type"] == "invalid_argument"
    assert "mesh_file_name" in out["error"]["message"]
    assert len(str(out)) < 1000


def test_the_config_never_receives_a_payload(tmp_path):
    cfg = tmp_path / "config.cfg"
    cfg.write_text("MESH_FILENAME= mesh.su2\nSOLVER= EULER\n")
    with pytest.raises(ValueError):
        config_utils.update_config_entries(cfg, {"MESH_FILENAME": HUGE})
    with pytest.raises(ValueError):
        config_utils.update_config_entries(cfg, {"SOME_OPTION": HUGE})
    assert cfg.read_text() == "MESH_FILENAME= mesh.su2\nSOLVER= EULER\n"


def test_the_log_parsers_are_fast_on_a_giant_line_and_still_find_real_errors():
    log = "\n".join([f"MESH_FILENAME= {HUGE}",
                     "Line 12 PHYSICAL_PROBLEM: invalid option name. Did you mean SOLVER?",
                     "MACH_NUMBER: invalid option value abc."])
    t0 = time.time()
    errors = parse_config_errors(log)
    parse_fatal_error(log)
    assert time.time() - t0 < 2.0                       # was ~800 s for this 400k line before the fix
    assert {e["option"] for e in errors} == {"PHYSICAL_PROBLEM", "MACH_NUMBER"}


def test_the_returned_log_tail_carries_no_giant_line():
    out = clip_log_lines(f"start\nMESH_FILENAME= {HUGE}\nend")
    assert len(out) < 1000 and out.startswith("start\n") and out.endswith("\nend")
    assert "characters cut" in out
