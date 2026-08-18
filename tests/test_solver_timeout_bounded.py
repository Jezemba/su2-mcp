"""The solver timeout must actually bound the call.

Measured 2026-08-17 (MAS-Aviary all7 run 5): run_su2_solver was given
max_runtime_seconds=600 and blocked for 131 MINUTES. The run died on the outer
150-minute wall clock having done 19 minutes of real work.

`subprocess.run(..., stdout=PIPE, timeout=N)` looks like it bounds the call and
does not: Python kills the direct child, but SU2 forks MPI ranks that keep the
write end of the pipe open, so the read never returns. A long solve can also fill
the pipe buffer and deadlock before any timeout fires.

These tests use a stand-in that reproduces the shape -- a process that forks a
child holding the output open and then sleeps far past the cap. Under the old
implementation this hangs; under the fix it returns within the timeout.
"""

import subprocess
import time
from pathlib import Path

import pytest

from su2_mcp.su2_runner import SU2Runner


@pytest.fixture
def runner(tmp_path):
    return SU2Runner(tmp_path)


def _fake_solver(tmp_path: Path, body: str) -> str:
    """A script standing in for SU2, marked executable."""
    p = tmp_path / "fake_solver.sh"
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(0o755)
    return str(p)


class TestTheTimeoutIsHonoured:
    def test_a_process_that_forks_and_sleeps_is_still_bounded(self, runner, tmp_path):
        """The exact failure shape: a child outlives the parent holding output."""
        solver = _fake_solver(tmp_path, "sleep 120 & echo started; sleep 120\n")
        cfg = tmp_path / "config.cfg"; cfg.write_text("SOLVER= EULER\n")

        start = time.time()
        result = runner.run(solver, cfg, max_runtime_seconds=3, capture_log_lines=10)
        elapsed = time.time() - start

        assert elapsed < 30, f"timeout not honoured: took {elapsed:.0f}s for a 3s cap"
        assert result["success"] is False

    def test_no_orphan_survives_the_timeout(self, runner, tmp_path):
        """A forked rank left alive would hold GPU/CPU and the output handle."""
        marker = tmp_path / "still_alive.txt"
        solver = _fake_solver(
            tmp_path, f"( sleep 25; echo leaked > {marker} ) & echo started; sleep 25\n"
        )
        cfg = tmp_path / "config.cfg"; cfg.write_text("SOLVER= EULER\n")
        runner.run(solver, cfg, max_runtime_seconds=2, capture_log_lines=10)
        time.sleep(6)
        assert not marker.exists(), "a forked child survived the group kill"


class TestNormalOperationIsUnchanged:
    def test_a_quick_success_still_reports_success(self, runner, tmp_path):
        solver = _fake_solver(tmp_path, "echo 'Exit Success'\nexit 0\n")
        cfg = tmp_path / "config.cfg"; cfg.write_text("SOLVER= EULER\n")
        r = runner.run(solver, cfg, max_runtime_seconds=30, capture_log_lines=10)
        assert r["success"] is True and r["exit_code"] == 0

    def test_the_log_tail_is_still_captured(self, runner, tmp_path):
        solver = _fake_solver(tmp_path, "echo 'rms[Rho] -6.01'\necho 'Exit Success'\nexit 0\n")
        cfg = tmp_path / "config.cfg"; cfg.write_text("SOLVER= EULER\n")
        r = runner.run(solver, cfg, max_runtime_seconds=30, capture_log_lines=10)
        assert "rms[Rho]" in r["log_tail"]

    def test_a_real_failure_still_reports_its_exit_code(self, runner, tmp_path):
        solver = _fake_solver(tmp_path, "echo 'Error Exit'\nexit 1\n")
        cfg = tmp_path / "config.cfg"; cfg.write_text("SOLVER= EULER\n")
        r = runner.run(solver, cfg, max_runtime_seconds=30, capture_log_lines=10)
        assert r["success"] is False and r["exit_code"] == 1
