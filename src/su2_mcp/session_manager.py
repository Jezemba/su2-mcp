"""Session management utilities for SU2 MCP server."""

from __future__ import annotations

import base64
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path


_MESH_NAME_MAX = 255
_MESH_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def check_mesh_file_name(mesh_file_name: object) -> str:
    """The mesh file name, or ValueError if it is not a plain file name (Avion B267).

    2026-11-10, l2c seq_graph: a caller passed the 155,514,712-character mesh itself as mesh_file_name.
    It was written into the session config as MESH_FILENAME, SU2 echoed it into its log, and the log
    parser spun on that one line with the server's only thread -- the server answered nothing for hours.
    A mesh file name is a short name in the session's workdir; anything else is refused before it is
    written anywhere, and the refusal never echoes the value back.
    """
    name = str(mesh_file_name if mesh_file_name is not None else "")
    if not name or len(name) > _MESH_NAME_MAX or not set(name) <= _MESH_NAME_CHARS or name in (".", ".."):
        shown = name if len(name) <= 60 else f"{name[:40]}... ({len(name):,} characters)"
        raise ValueError(
            f"mesh_file_name must be a plain file name such as 'mesh.su2' (letters, digits, '.', '_', '-'; "
            f"at most {_MESH_NAME_MAX} characters, no path); got {shown!r}. The mesh CONTENTS go in "
            "initial_mesh / mesh_base64, never in mesh_file_name. Nothing was written."
        )
    return name


def clip_text(text: object, limit: int = 2000) -> str:
    """At most *limit* characters of *text*, saying how much was cut (B267: no 155 MB error replies)."""
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text) - limit:,} more characters cut]"


def decode_mesh(mesh_base64: str, mesh_file_name: str) -> bytes:
    """Decode a base64 mesh payload, refusing content that is not a mesh.

    ``base64.b64decode`` accepts far more than it should here. Every one of
    these stand-ins was passed as ``mesh_base64`` during the 2026-08 sweeps,
    and three of the five decoded WITHOUT error --

        'exported'            -> 6 bytes    accepted
        'base64_mesh_data=='  -> 10 bytes   accepted
        'SGVsbG8gd29ybGQ='    -> 11 bytes   accepted  ("Hello world")
        '<MESH_BASE64...>'    -> rejected   (incorrect padding)
        'mesh.su2'            -> rejected   (incorrect padding)

    -- so the bytes were written to mesh.su2 and the caller was told the mesh
    was SET. The real failure surfaced later and elsewhere, as SU2 complaining
    about a mesh the caller believed it had just supplied successfully. A wrong
    answer reported as success is worse than an error, because nothing
    downstream has any reason to doubt it.

    The SU2 native format declares its dimensionality first (``NDIME=``; a real
    export here begins ``NDIME= 3``), so for .su2 the check is the format's own
    requirement rather than a size threshold. Other formats (e.g. CGNS) are
    passed through undecoded-content-checked -- we validate only what we can
    identify, instead of guessing a minimum size.
    """
    mesh_bytes = base64.b64decode(mesh_base64)
    if not str(mesh_file_name).lower().endswith(".su2"):
        return mesh_bytes
    if b"NDIME" in mesh_bytes[:512]:
        return mesh_bytes
    preview = mesh_bytes[:40].decode("utf-8", "replace")
    raise ValueError(
        f"mesh_base64 decoded to {len(mesh_bytes)} bytes that are not an SU2 mesh: "
        f"a .su2 file must declare NDIME= in its header, and this begins {preview!r}. "
        "The mesh was NOT stored. This usually means a name, a path, or a stand-in "
        "token was passed where the base64-encoded mesh FILE CONTENTS belong; take "
        "them from the meshing step that produced the mesh."
    )


@dataclass
class LastRunMetadata:
    """Metadata captured after running a SU2 solver."""

    solver: str
    config_used: str
    exit_code: int
    runtime_seconds: float
    log_tail: str


@dataclass
class SessionRecord:
    """Session state stored by :class:`SessionManager`."""

    session_id: str
    workdir: Path
    config_path: Path
    mesh_path: Path | None = None
    last_run_metadata: LastRunMetadata | None = None
    # For the induced-drag floor on the reported CD (drag_model.assess_drag, Avion B138).
    aspect_ratio: float | None = None
    oswald_efficiency: float | None = None
    # Option names SU2 rejected as invalid in this session (B173); configure_from_cpacs removes them.
    invalid_options: set = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    """Manage SU2 sessions and their resources."""

    def __init__(self) -> None:
        """Create an empty session registry."""
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        base_name: str | None = None,
        initial_config: str | None = None,
        initial_mesh: str | None = None,
        mesh_file_name: str = "mesh.su2",
    ) -> SessionRecord:
        """Create a new SU2 session with optional seed files."""
        mesh_file_name = check_mesh_file_name(mesh_file_name)   # B267: before anything is written
        session_id = str(uuid.uuid4())
        prefix = f"su2_{base_name}_" if base_name else "su2_session_"
        workdir = Path(tempfile.mkdtemp(prefix=prefix))
        config_path = workdir / "config.cfg"

        config_text = (
            initial_config or "% Minimal SU2 config\nMESH_FILENAME= mesh.su2\n"
        )
        config_path.write_text(config_text)

        mesh_path: Path | None = None
        if initial_mesh is not None:
            mesh_bytes = decode_mesh(initial_mesh, mesh_file_name)
            mesh_path = workdir / mesh_file_name
            mesh_path.write_bytes(mesh_bytes)
            self._ensure_mesh_filename_in_config(config_path, mesh_file_name)

        record = SessionRecord(
            session_id=session_id,
            workdir=workdir,
            config_path=config_path,
            mesh_path=mesh_path,
        )

        with self._lock:
            self._sessions[session_id] = record

        return record

    def _ensure_mesh_filename_in_config(
        self, config_path: Path, mesh_file_name: str
    ) -> None:
        """Ensure the configuration declares the mesh filename."""
        mesh_file_name = check_mesh_file_name(mesh_file_name)
        existing = config_path.read_text().splitlines()
        updated = []
        mesh_key_written = False
        for line in existing:
            if line.strip().startswith("MESH_FILENAME"):
                updated.append(f"MESH_FILENAME= {mesh_file_name}")
                mesh_key_written = True
            else:
                updated.append(line)
        if not mesh_key_written:
            updated.append(f"MESH_FILENAME= {mesh_file_name}")
        config_path.write_text("\n".join(updated) + "\n")

    def close_session(self, session_id: str, delete_workdir: bool = False) -> bool:
        """Close a session and optionally remove its working directory."""
        with self._lock:
            record = self._sessions.pop(session_id, None)
        if record is None:
            return False
        if delete_workdir:
            shutil.rmtree(record.workdir, ignore_errors=True)
        return True

    def get(self, session_id: str) -> SessionRecord | None:
        """Return the session record if it exists."""
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> SessionRecord:
        """Return the session record or raise if missing."""
        record = self.get(session_id)
        if record is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return record

    def update_mesh(
        self,
        session_id: str,
        mesh_base64: str,
        mesh_file_name: str = "mesh.su2",
    ) -> Path:
        """Persist a mesh to the session directory and update bookkeeping."""
        mesh_file_name = check_mesh_file_name(mesh_file_name)
        record = self.require(session_id)
        mesh_bytes = decode_mesh(mesh_base64, mesh_file_name)
        mesh_path = record.workdir / mesh_file_name
        mesh_path.write_bytes(mesh_bytes)
        record.mesh_path = mesh_path
        self._ensure_mesh_filename_in_config(record.config_path, mesh_file_name)
        return mesh_path

    def to_info(self, session_id: str) -> dict[str, object]:
        """Return a JSON-friendly description of the session."""
        record = self.require(session_id)
        return {
            "session_id": record.session_id,
            "workdir": str(record.workdir),
            "config_path": str(record.config_path),
            "mesh_path": str(record.mesh_path) if record.mesh_path else None,
            "last_run": (
                None
                if record.last_run_metadata is None
                else {
                    "solver": record.last_run_metadata.solver,
                    "config_used": record.last_run_metadata.config_used,
                    "exit_code": record.last_run_metadata.exit_code,
                    "runtime_seconds": record.last_run_metadata.runtime_seconds,
                    "log_tail": record.last_run_metadata.log_tail,
                }
            ),
        }

    def record_run(self, session_id: str, metadata: LastRunMetadata) -> None:
        """Store the last run metadata for the session."""
        record = self.require(session_id)
        record.last_run_metadata = metadata
