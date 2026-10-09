"""SU2_MPI_RANKS=auto takes only the free cores (Avion, 2026-10-09: aipexws5 is shared)."""
import pytest

from su2_mcp import mpi


@pytest.fixture
def mpi_build(monkeypatch):
    monkeypatch.setattr(mpi.shutil, "which", lambda n: f"/opt/su2/{n}" if n in ("SU2_CFD", "mpirun") else None)
    monkeypatch.setattr(mpi, "_links_against_mpi", lambda exe: True)
    monkeypatch.setattr(mpi, "_launcher_flavour", lambda launcher: "openmpi")
    monkeypatch.setattr(mpi, "_physical_cores", lambda: 24)
    monkeypatch.setenv("SU2_MPI_RANKS", "auto")
    for k in ("SU2_MPI_MAX_RANKS", "SU2_MPI_FREE_CORES", "SU2_MPIRUN_ARGS"):
        monkeypatch.delenv(k, raising=False)


# a partly busy core counts as busy
@pytest.mark.parametrize("load,expected", [(0.6, 19), (10.2, 9), (19.5, 1), (40.0, 1)])
def test_auto_leaves_busy_cores_and_a_reserve(mpi_build, monkeypatch, load, expected):
    monkeypatch.setattr(mpi.os, "getloadavg", lambda: (load, load, load))
    d = mpi.parallel_decision("SU2_CFD")
    assert d["ranks"] == expected and d["command"][-2:] == [str(expected), "/opt/su2/SU2_CFD"]
    assert "load-aware" in d["reason"]


def test_the_cap_and_the_reserve_are_settable(mpi_build, monkeypatch):
    monkeypatch.setattr(mpi.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setenv("SU2_MPI_MAX_RANKS", "16")
    assert mpi.parallel_decision("SU2_CFD")["ranks"] == 16
    monkeypatch.setenv("SU2_MPI_MAX_RANKS", "")
    monkeypatch.setenv("SU2_MPI_FREE_CORES", "8")
    assert mpi.parallel_decision("SU2_CFD")["ranks"] == 16


def test_without_auto_the_behaviour_is_unchanged(mpi_build, monkeypatch):
    monkeypatch.setattr(mpi.os, "getloadavg", lambda: (20.0, 20.0, 20.0))
    monkeypatch.setenv("SU2_MPI_RANKS", "12")
    assert mpi.parallel_decision("SU2_CFD")["ranks"] == 12
    monkeypatch.delenv("SU2_MPI_RANKS")
    assert mpi.parallel_decision("SU2_CFD")["ranks"] == 24
