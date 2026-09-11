"""Custom KNIME port type for KSurfer SSH connection.

Cross-process credential sharing via temp file
-----------------------------------------------
KNIME runs each node in a separate Python process. A module-level dict
cannot be shared between processes. Instead we use a temporary JSON file
in the system temp directory, keyed by session ID, to pass credentials
from the SSH Connector to downstream nodes.

The temp file stores host, port, username, auth_method, key_path, and a
base64-encoded password. Each downstream node reads the file and opens
its own short-lived paramiko connection without needing a password field
in its dialog. The file is owner-readable only (chmod 600).
"""

import os
import json
import uuid
import base64
import tempfile
import logging

import knime.extension as knext

LOGGER = logging.getLogger(__name__)

_SESSION_FILE_PREFIX = "ksurfer_session_"


def _session_file_path(session_id: str) -> str:
    return os.path.join(
        tempfile.gettempdir(),
        f"{_SESSION_FILE_PREFIX}{session_id}.json"
    )


def store_session(
    session_id: str,
    host: str,
    port: int,
    username: str,
    auth_method: str,
    key_path: str,
    password: str,
) -> None:
    """Write session credentials to a temp file for downstream nodes."""
    data = {
        "session_id":   session_id,
        "host":         host,
        "port":         port,
        "username":     username,
        "auth_method":  auth_method,
        "key_path":     key_path,
        "password_b64": base64.b64encode(
            (password or "").encode()
        ).decode(),
    }
    path = _session_file_path(session_id)
    with open(path, "w") as f:
        json.dump(data, f)
    os.chmod(path, 0o600)
    LOGGER.warning(
        "KSurfer: session credentials stored [%s]", session_id[:8]
    )


def load_session(session_id: str) -> dict:
    """Read session credentials from the temp file."""
    path = _session_file_path(session_id)
    if not os.path.exists(path):
        raise RuntimeError(
            "SSH session credentials not found. The session may have "
            "expired or the KSurfer SSH Connector has not been executed.\n"
            "Please re-execute the KSurfer SSH Connector node."
        )
    with open(path) as f:
        data = json.load(f)
    data["password"] = base64.b64decode(
        data.pop("password_b64", "")
    ).decode()
    return data


def delete_session(session_id: str) -> None:
    """Remove the session temp file."""
    path = _session_file_path(session_id)
    try:
        os.remove(path)
    except OSError:
        pass


def open_client_from_session(session_id: str):
    """Open a fresh paramiko client using stored session credentials."""
    from ksurfer_core.ssh import open_client
    data = load_session(session_id)
    return open_client(
        host=data["host"],
        port=data["port"],
        username=data["username"],
        auth_method=data["auth_method"],
        key_path=data.get("key_path", ""),
        password=data["password"] or None,
    )


# ---------------------------------------------------------------------------
# Port type
# ---------------------------------------------------------------------------

class KSurferConnectionSpec(knext.PortObjectSpec):
    """Carries session ID and connection metadata between KSurfer nodes."""

    def __init__(
        self,
        session_id: str,
        host: str,
        port: int,
        username: str,
        auth_method: str,
        key_path: str,
    ):
        super().__init__()
        self._session_id  = session_id
        self._host        = host
        self._port        = port
        self._username    = username
        self._auth_method = auth_method
        self._key_path    = key_path

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def username(self) -> str:
        return self._username

    @property
    def auth_method(self) -> str:
        return self._auth_method

    @property
    def key_path(self) -> str:
        return self._key_path

    def open_client(self):
        """Open a fresh paramiko client using stored session credentials."""
        return open_client_from_session(self._session_id)

    def serialize(self) -> dict:
        return {
            "session_id":   self._session_id,
            "host":         self._host,
            "port":         self._port,
            "username":     self._username,
            "auth_method":  self._auth_method,
            "key_path":     self._key_path,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "KSurferConnectionSpec":
        return cls(
            session_id=data["session_id"],
            host=data["host"],
            port=data["port"],
            username=data["username"],
            auth_method=data["auth_method"],
            key_path=data.get("key_path", ""),
        )


class KSurferConnectionPortObject(knext.PortObject):
    def __init__(self, spec: KSurferConnectionSpec):
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(
        cls, spec: KSurferConnectionSpec, storage: bytes
    ) -> "KSurferConnectionPortObject":
        return cls(spec)


ksurfer_connection_port_type = knext.port_type(
    name="KSurfer SSH Connection",
    object_class=KSurferConnectionPortObject,
    spec_class=KSurferConnectionSpec,
)
