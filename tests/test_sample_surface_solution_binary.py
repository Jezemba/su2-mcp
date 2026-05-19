"""Regression test for sample_surface_solution against binary inputs.

The Avion MDO pipeline (Run #11, 2026-05-18) hit a cryptic error when
the aero agent passed ``relative_path="surface.vtu"`` to
``sample_surface_solution``::

    {"error": {"type": "runtime_error",
               "message": "Failed to sample surface solution",
               "details": "'utf-8' codec can't decode byte 0x94 in
                           position 1487: invalid start byte"}}

The tool was opening any caller-supplied file with
``encoding="utf-8"`` and feeding it to ``csv.DictReader`` — fine for
``surface_flow.csv`` but explosively wrong for the binary ``.vtu``
that SU2 also writes. The contract this test pins down is: when
handed a non-text file, the tool MUST return a structured
``validation_error`` (not raise UnicodeDecodeError) and the message
must tell the caller to pass a CSV path instead.
"""

from __future__ import annotations

from su2_mcp.tools import results_tools, session


def test_sample_surface_solution_rejects_binary_vtu() -> None:
    """Binary surface.vtu input returns a clear validation_error, not a
    UTF-8 decode crash."""
    created = session.create_su2_session()
    session_id = str(created["session_id"])
    record = session.SESSION_MANAGER.require(session_id)

    # Mimic the file SU2 actually writes: a VTK XML header followed by
    # appended binary data including a byte (0x94) that's invalid UTF-8.
    vtu = record.workdir / "surface.vtu"
    vtu.write_bytes(
        b'<?xml version="1.0"?>\n'
        b'<VTKFile type="UnstructuredGrid" byte_order="LittleEndian">\n'
        b'<AppendedData encoding="raw">_\x94\xff\x00binary_bytes_here'
    )

    result = results_tools.sample_surface_solution(
        session_id, "surface.vtu", marker_name="aircraft", fields=["Cp"]
    )

    assert "error" in result, (
        f"Expected structured error, got: {result!r}"
    )
    assert result["error"]["type"] == "validation_error", (
        f"Expected validation_error type, got: {result['error']!r}"
    )
    # The message should mention CSV so the agent knows how to recover.
    message_blob = (
        result["error"].get("message", "")
        + " "
        + str(result["error"].get("details", ""))
    ).lower()
    assert (
        "csv" in message_blob
        or "binary" in message_blob
        or "surface_flow" in message_blob
    ), (
        "Error message should steer the caller toward a CSV input; got: "
        f"{result['error']!r}"
    )

    session.close_su2_session(session_id, delete_workdir=True)


def test_sample_surface_solution_accepts_csv_with_binary_extension_guess() -> None:
    """Files with a CSV extension must still be parsed as CSV even if
    they begin with bytes that look unusual."""
    created = session.create_su2_session()
    session_id = str(created["session_id"])
    record = session.SESSION_MANAGER.require(session_id)

    # Plain CSV — should parse normally and return the row data.
    surface_csv = record.workdir / "surface_flow.csv"
    surface_csv.write_text(
        "marker,Cp,Cf\nWALL,0.5,0.001\nWALL,-0.2,0.0008\n",
        encoding="utf-8",
    )

    result = results_tools.sample_surface_solution(
        session_id,
        "surface_flow.csv",
        marker_name="WALL",
        fields=["Cp", "Cf"],
        max_points=10,
    )
    assert "error" not in result, f"Unexpected error: {result!r}"
    assert result["num_points"] == 2
    assert result["points"][0] == {"Cp": 0.5, "Cf": 0.001}

    session.close_su2_session(session_id, delete_workdir=True)
