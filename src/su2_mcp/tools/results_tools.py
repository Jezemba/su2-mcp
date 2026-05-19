"""Result inspection MCP tools."""

from __future__ import annotations

import base64
import csv
import datetime as dt

from su2_mcp.tools.session import SESSION_MANAGER, _error


def list_result_files(
    session_id: str, extensions: list[str] | None = None
) -> dict[str, object]:
    """List result files produced within a session directory."""
    try:
        record = SESSION_MANAGER.require(session_id)
        files: list[dict[str, object]] = []
        for path in record.workdir.rglob("*"):
            if path.is_file():
                if extensions and path.suffix not in extensions:
                    continue
                files.append(
                    {
                        "path": str(path.relative_to(record.workdir)),
                        "size_bytes": path.stat().st_size,
                        "modified_time": dt.datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).isoformat(),
                    }
                )
        return {"files": files}
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to list result files", details=str(exc))


def get_result_file_base64(
    session_id: str, relative_path: str, max_bytes: int = 104_857_600
) -> dict[str, object]:
    """Return a base64-encoded slice of a result file."""
    try:
        record = SESSION_MANAGER.require(session_id)
        workdir = record.workdir.resolve()
        full_path = (workdir / relative_path).resolve()
        if workdir not in full_path.parents and workdir != full_path:
            return _error(
                "Requested path outside workdir", error_type="validation_error"
            )
        if not full_path.exists():
            return _error("Result file not found", error_type="not_found")
        size_bytes = full_path.stat().st_size
        truncated = size_bytes > max_bytes
        data = full_path.read_bytes()[:max_bytes]
        return {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "truncated": truncated,
            "data_base64": base64.b64encode(data).decode("utf-8"),
        }
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to read result file", details=str(exc))


def read_history_csv(
    session_id: str,
    relative_path: str,
    columns: list[str] | None = None,
    max_rows: int = 1000,
    skip_rows: int = 0,
) -> dict[str, object]:
    """Read a subset of a history CSV file."""
    try:
        record = SESSION_MANAGER.require(session_id)
        full_path = record.workdir / relative_path
        if not full_path.exists():
            return _error("History file not found", error_type="not_found")
        with full_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            filtered_headers = columns or headers
            rows: list[dict[str, object]] = []
            for _ in range(skip_rows):
                next(reader, None)
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                rows.append(
                    {
                        key: _coerce_value(row.get(key))
                        for key in filtered_headers
                        if key in row
                    }
                )
        return {
            "columns": filtered_headers,
            "rows": rows,
            "total_rows": skip_rows + len(rows),
        }
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to parse history CSV", details=str(exc))


def _coerce_value(value: str | None) -> object | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


_BINARY_SURFACE_HINTS = (".vtu", ".vtk", ".dat", ".plt", ".bin", ".szplt")


def sample_surface_solution(
    session_id: str,
    relative_path: str,
    marker_name: str | None,
    fields: list[str],
    max_points: int = 5000,
) -> dict[str, object]:
    """Sample surface solution fields from a CSV-like file.

    The caller MUST point at a text/CSV surface solution (e.g.
    ``surface_flow.csv``, written when SU2's ``OUTPUT_FILES`` includes
    ``SURFACE_CSV``). Binary files such as the ``surface.vtu`` ParaView
    output are explicitly rejected — that path used to crash with a
    cryptic ``'utf-8' codec can't decode byte ...`` error.
    """
    try:
        record = SESSION_MANAGER.require(session_id)
        full_path = record.workdir / relative_path
        if not full_path.exists():
            return _error("Surface solution not found", error_type="not_found")

        # Reject obviously-binary surface formats before opening the file —
        # the message tells the agent where the CSV actually lives.
        suffix = full_path.suffix.lower()
        if suffix in _BINARY_SURFACE_HINTS:
            return _error(
                "Surface solution must be a text CSV (e.g. "
                "'surface_flow.csv'), not a binary file like "
                f"'{full_path.name}'. Add SURFACE_CSV to SU2's "
                "OUTPUT_FILES to emit the CSV.",
                error_type="validation_error",
            )

        try:
            with full_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows: list[dict[str, object]] = []
                for idx, row in enumerate(reader):
                    if idx >= max_points:
                        break
                    filtered = {
                        field: _coerce_value(row.get(field))
                        for field in fields
                        if field in row
                    }
                    if (
                        marker_name
                        and row.get("marker")
                        and row.get("marker") != marker_name
                    ):
                        continue
                    rows.append(filtered)
        except UnicodeDecodeError as decode_exc:
            return _error(
                "Surface solution is not valid UTF-8 text; this tool "
                "only handles CSV. Re-run SU2 with SURFACE_CSV in "
                "OUTPUT_FILES and read 'surface_flow.csv' instead.",
                error_type="validation_error",
                details=str(decode_exc),
            )

        return {"marker_name": marker_name, "points": rows, "num_points": len(rows)}
    except KeyError as exc:
        return _error(str(exc), error_type="not_found")
    except Exception as exc:  # pragma: no cover
        return _error("Failed to sample surface solution", details=str(exc))


__all__ = [
    "list_result_files",
    "get_result_file_base64",
    "read_history_csv",
    "sample_surface_solution",
]
