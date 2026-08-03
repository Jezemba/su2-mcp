"""read_history_csv must not silently return empty rows.

Found live during the 2026-08-03 MAS-Aviary sweep. An agent called

    read_history_csv(..., columns=['ITER','RMS_RES','LIFT','DRAG'], ...)

and got back 75 rows, every one an empty dict, with `columns` echoing the
requested names as if they existed. The mission then flew UNCOUPLED even though
SU2 had converged and written CL/CD correctly.

Two causes, both fixed here:
  1. SU2 pads and quotes its headers ('       "CD"       '), so raw equality
     against a caller's "CD" never matched -- the column filter was effectively
     unusable against real SU2 history files.
  2. Unknown/unmatched columns were dropped silently instead of reported.
"""

import csv

import pytest

from su2_mcp.tools.results_tools import read_history_csv

# Verbatim header shape from a real SU2 8.3 history.csv.
SU2_HEADERS = [
    "Time_Iter", "Outer_Iter", "Inner_Iter",
    '    "rms[Rho]"    ', '    "rms[RhoU]"   ',
    '       "CD"       ', '       "CL"       ', '       "CSF"      ',
]


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A stub session whose workdir holds a realistic history.csv."""
    hist = tmp_path / "history.csv"
    with hist.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SU2_HEADERS)
        for i in range(3):
            w.writerow([0.0, 0.0, float(i), -0.03 - i, 2.3, 0.0018375, 0.09221, -0.0079])

    class _Rec:
        workdir = tmp_path

    class _Mgr:
        def require(self, session_id):
            return _Rec()

    monkeypatch.setattr("su2_mcp.tools.results_tools.SESSION_MANAGER", _Mgr())
    return "sid"


class TestPaddedQuotedHeaders:
    def test_plain_cl_cd_now_resolve(self, session):
        """THE regression: asking for CL/CD must match '   "CD"   '."""
        out = read_history_csv(session, "history.csv", columns=["CL", "CD"])
        assert "unmatched_columns" not in out
        assert out["rows"], "rows must not be empty"
        assert out["rows"][-1]["CL"] == pytest.approx(0.09221)
        assert out["rows"][-1]["CD"] == pytest.approx(0.0018375)

    def test_rows_are_not_empty_dicts(self, session):
        out = read_history_csv(session, "history.csv", columns=["CL", "CD"])
        assert all(r for r in out["rows"]), "no row may be an empty dict"

    def test_quoted_request_also_resolves(self, session):
        out = read_history_csv(session, "history.csv", columns=['  "CL"  '])
        assert out["rows"][-1]['  "CL"  '] == pytest.approx(0.09221)


class TestUnmatchedColumnsReported:
    def test_history_output_group_names_are_flagged(self, session):
        """The exact live failure: LIFT/DRAG are GROUP names, not columns."""
        out = read_history_csv(
            session, "history.csv", columns=["ITER", "RMS_RES", "LIFT", "DRAG"]
        )
        assert set(out["unmatched_columns"]) == {"ITER", "RMS_RES", "LIFT", "DRAG"}
        assert "CL" in out["available_columns"]
        assert "CD" in out["available_columns"]
        assert "not in this history file" in out["warning"]

    def test_partial_match_reports_only_the_bad_ones(self, session):
        out = read_history_csv(session, "history.csv", columns=["CL", "NOPE"])
        assert out["unmatched_columns"] == ["NOPE"]
        assert out["rows"][-1]["CL"] == pytest.approx(0.09221)
        assert "NOPE" not in out["rows"][-1]

    def test_available_columns_are_normalised(self, session):
        out = read_history_csv(session, "history.csv", columns=["NOPE"])
        assert '    "rms[Rho]"    ' not in out["available_columns"]
        assert "rms[Rho]" in out["available_columns"]


class TestNoFilterUnaffected:
    def test_all_columns_returned_when_no_filter(self, session):
        out = read_history_csv(session, "history.csv")
        assert "unmatched_columns" not in out
        assert len(out["rows"]) == 3
        assert len(out["columns"]) == len(SU2_HEADERS)

    def test_total_rows_and_skip_rows(self, session):
        out = read_history_csv(session, "history.csv", columns=["CL"], skip_rows=1)
        assert len(out["rows"]) == 2
        assert out["total_rows"] == 3
