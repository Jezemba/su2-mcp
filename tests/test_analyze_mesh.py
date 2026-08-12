"""Coverage for the analyze_mesh tool."""

from __future__ import annotations

from su2_mcp.tools import mesh_tools, session


def test_analyze_mesh_missing_session() -> None:
    """Unknown session IDs return not_found."""
    result = mesh_tools.analyze_mesh("missing")
    assert result["error"]["type"] == "not_found"


def test_analyze_mesh_no_mesh_file() -> None:
    """Sessions without a mesh file return not_found."""
    created = session.create_su2_session()
    session_id = str(created["session_id"])

    result = mesh_tools.analyze_mesh(session_id)
    assert result["error"]["type"] == "not_found"

    session.close_su2_session(session_id, delete_workdir=True)


def test_analyze_mesh_reads_a_real_mesh() -> None:
    """The success path was never exercised, so an AttributeError on every call
    -- `record.mesh_filename` does not exist; the field is `mesh_path` -- went
    unnoticed. Both existing tests returned before reaching it.
    """
    import base64

    from su2_mcp.tools import config_tools

    mesh = (
        "NDIME= 2\n"
        "NPOIN= 4\n0.0 0.0 0\n1.0 0.0 1\n1.0 1.0 2\n0.0 1.0 3\n"
        "NELEM= 1\n9 0 1 2 3 0\n"
        "NMARK= 1\nMARKER_TAG= airfoil\nMARKER_ELEMS= 1\n3 0 1\n"
    )
    created = session.create_su2_session()
    session_id = str(created["session_id"])
    try:
        stored = config_tools.set_mesh(
            session_id, base64.b64encode(mesh.encode()).decode()
        )
        assert "error" not in stored, stored

        result = mesh_tools.analyze_mesh(session_id)
        assert "error" not in result, result
        assert result["mesh_file"] == "mesh.su2"
        assert result["file_size_bytes"] > 0
    finally:
        session.close_su2_session(session_id, delete_workdir=True)
