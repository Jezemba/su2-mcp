"""Guarantee an SU2 run will actually write force coefficients.

Motivating failure (real MAS-Aviary networked run, 2026-08-02): the agent ran
the solver, read the history, and re-applied the mission parameters — the whole
coupling contract — yet the mission silently flew on aviary's default drag
polar. history.csv contained only

    Time_Iter, Outer_Iter, Inner_Iter, "rms[Rho]", "rms[RhoU]", ...

because the config had no MARKER_MONITORING. SU2 integrates forces ONLY over
monitored markers, so there were never any CL/CD to capture. The aero stage
"succeeded" while being physically incapable of coupling.
"""

import pytest

from su2_mcp.config_utils import ensure_force_output, parse_config_file

MINIMAL_NO_FORCES = """\
SOLVER= EULER
MACH_NUMBER= 0.78
AOA= 2.0
MESH_FILENAME= mesh.su2
MARKER_FAR= ( farfield )
MARKER_EULER= ( aircraft )
ITER= 250
CONV_FILENAME= history
HISTORY_OUTPUT= ( ITER, RMS_RES )
"""

FULLY_CONFIGURED = """\
SOLVER= EULER
MARKER_FAR= ( farfield )
MARKER_EULER= ( aircraft )
MARKER_MONITORING= ( aircraft )
HISTORY_OUTPUT= ( ITER, RMS_RES, LIFT, DRAG, AERO_COEFF )
REF_AREA= 192.193
"""


def _write(tmp_path, text):
    p = tmp_path / "su2.cfg"
    p.write_text(text)
    return p


class TestMarkerMonitoring:
    def test_derived_from_marker_euler_when_missing(self, tmp_path):
        cfg = _write(tmp_path, MINIMAL_NO_FORCES)
        report = ensure_force_output(cfg)
        entries = parse_config_file(cfg)
        assert "MARKER_MONITORING" in report["added"]
        assert "aircraft" in str(entries["MARKER_MONITORING"])
        assert report["force_output_ok"] is True

    def test_derived_from_plotting_when_no_euler(self, tmp_path):
        cfg = _write(tmp_path, "MARKER_PLOTTING= ( wing )\nHISTORY_OUTPUT= ( ITER )\n")
        ensure_force_output(cfg)
        assert "wing" in str(parse_config_file(cfg)["MARKER_MONITORING"])

    def test_existing_monitoring_is_preserved(self, tmp_path):
        cfg = _write(tmp_path, "MARKER_EULER= ( aircraft )\nMARKER_MONITORING= ( custom_wall )\n")
        report = ensure_force_output(cfg)
        assert "MARKER_MONITORING" not in report["added"]
        assert "custom_wall" in str(parse_config_file(cfg)["MARKER_MONITORING"])

    def test_no_wall_marker_is_reported_not_silent(self, tmp_path):
        """The genuinely unfixable case must be LOUD, not quiet."""
        cfg = _write(tmp_path, "SOLVER= EULER\nMARKER_FAR= ( farfield )\n")
        report = ensure_force_output(cfg)
        assert report["force_output_ok"] is False
        assert any("CANNOT produce force" in n for n in report["notes"])


class TestHistoryOutput:
    def test_force_fields_appended_to_existing(self, tmp_path):
        cfg = _write(tmp_path, MINIMAL_NO_FORCES)
        ensure_force_output(cfg)
        hist = str(parse_config_file(cfg)["HISTORY_OUTPUT"]).upper()
        # Existing fields preserved, AERO_COEFF added.
        for field in ("ITER", "RMS_RES", "AERO_COEFF"):
            assert field in hist

    def test_no_duplicate_force_groups_added(self, tmp_path):
        """AERO_COEFF already expands to CL/CD — adding LIFT/DRAG too makes SU2
        emit DUPLICATE "CD","CD","CL","CL" columns (observed live)."""
        cfg = _write(tmp_path, MINIMAL_NO_FORCES)
        ensure_force_output(cfg)
        hist = str(parse_config_file(cfg)["HISTORY_OUTPUT"]).upper()
        assert "LIFT" not in hist
        assert "DRAG" not in hist

    def test_created_when_absent(self, tmp_path):
        cfg = _write(tmp_path, "MARKER_EULER= ( aircraft )\n")
        report = ensure_force_output(cfg)
        assert "HISTORY_OUTPUT" in report["added"]
        hist = str(parse_config_file(cfg)["HISTORY_OUTPUT"]).upper()
        assert "LIFT" in hist and "DRAG" in hist

    def test_preexisting_user_fields_are_kept(self, tmp_path):
        cfg = _write(tmp_path, "MARKER_EULER= ( a )\nHISTORY_OUTPUT= ( ITER, RMS_RES, LINSOL )\n")
        ensure_force_output(cfg)
        hist = str(parse_config_file(cfg)["HISTORY_OUTPUT"]).upper()
        assert "LINSOL" in hist and "AERO_COEFF" in hist

    def test_author_supplied_lift_drag_are_left_alone(self, tmp_path):
        """We only control what we ADD — an author's own LIFT/DRAG stay."""
        cfg = _write(tmp_path, "MARKER_EULER= ( a )\nHISTORY_OUTPUT= ( ITER, LIFT, DRAG )\n")
        ensure_force_output(cfg)
        hist = str(parse_config_file(cfg)["HISTORY_OUTPUT"]).upper()
        assert "LIFT" in hist and "DRAG" in hist and "AERO_COEFF" in hist


class TestReferenceArea:
    """Verified against official SU2 v8.3.0:
      CConfig.cpp:1488            addDoubleOption("REF_AREA", RefArea, 1.0)
      CPhysicalGeometry.cpp:4363  if (RefArea == 0.0) SetRefArea(TotalPositiveZArea)
    So 1.0 is the (meaningless-for-aircraft) default and 0 means AUTO-COMPUTE.
    """

    def test_missing_ref_area_is_set_to_auto(self, tmp_path):
        cfg = _write(tmp_path, MINIMAL_NO_FORCES)
        report = ensure_force_output(cfg)
        assert "REF_AREA" in report["added"]
        assert float(parse_config_file(cfg)["REF_AREA"]) == 0.0
        assert report["coefficients_meaningful"] is True

    def test_zero_is_valid_not_flagged(self, tmp_path):
        """0 is AUTO — the best setting. It must not be condemned."""
        cfg = _write(tmp_path, MINIMAL_NO_FORCES + "REF_AREA= 0\n")
        report = ensure_force_output(cfg)
        assert "REF_AREA" not in report["added"]
        assert report["coefficients_meaningful"] is True

    def test_exactly_one_is_flagged_but_not_overridden(self, tmp_path):
        """SU2's default leaking through — warn, but never rewrite an explicit
        value (a unit-chord 2D case is legitimate)."""
        cfg = _write(tmp_path, MINIMAL_NO_FORCES + "REF_AREA= 1.0\n")
        report = ensure_force_output(cfg)
        assert report["coefficients_meaningful"] is False
        assert any("exactly 1.0" in n for n in report["notes"])
        assert float(parse_config_file(cfg)["REF_AREA"]) == 1.0

    def test_real_area_is_left_alone(self, tmp_path):
        cfg = _write(tmp_path, MINIMAL_NO_FORCES + "REF_AREA= 192.193\n")
        report = ensure_force_output(cfg)
        assert "REF_AREA" not in report["added"]
        assert report["coefficients_meaningful"] is True
        assert float(parse_config_file(cfg)["REF_AREA"]) == pytest.approx(192.193)


class TestNonDestructive:
    def test_correct_config_is_untouched(self, tmp_path):
        cfg = _write(tmp_path, FULLY_CONFIGURED)
        report = ensure_force_output(cfg)
        assert report["added"] == []
        assert report["force_output_ok"] is True

    def test_unrelated_settings_survive(self, tmp_path):
        cfg = _write(tmp_path, MINIMAL_NO_FORCES)
        ensure_force_output(cfg)
        entries = parse_config_file(cfg)
        assert entries["SOLVER"] == "EULER"
        assert entries["MACH_NUMBER"] == pytest.approx(0.78)
        assert entries["AOA"] == pytest.approx(2.0)
        assert entries["ITER"] == 250
        assert entries["MESH_FILENAME"] == "mesh.su2"

    def test_idempotent(self, tmp_path):
        cfg = _write(tmp_path, MINIMAL_NO_FORCES)
        ensure_force_output(cfg)
        first = cfg.read_text()
        second_report = ensure_force_output(cfg)
        assert second_report["added"] == []
        assert cfg.read_text() == first
