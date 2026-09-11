"""SSH / SFTP helpers for the KSurfer extraction pipeline.

Depends on paramiko and pandas only — no KNIME dependency.
"""

from __future__ import annotations

import io
import logging
import posixpath
import shlex
from urllib.parse import urlparse

import pandas as pd
import paramiko

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def open_client(
    host: str,
    port: int,
    username: str,
    auth_method: str,
    key_path: str,
    password: str | None,
    accept_new_host_key: bool = False,
) -> paramiko.SSHClient:
    """Open and return an authenticated paramiko SSHClient."""
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    policy = paramiko.AutoAddPolicy() if accept_new_host_key else paramiko.RejectPolicy()
    client.set_missing_host_key_policy(policy)

    kw: dict = dict(hostname=host, port=port, username=username, timeout=15)
    if auth_method == "key":
        kw["key_filename"] = key_path
        if password:
            kw["passphrase"] = password
    else:
        kw["password"] = password

    try:
        client.connect(**kw)
    except paramiko.SSHException as exc:
        raise RuntimeError(
            f"SSH connection to {username}@{host}:{port} failed: {exc}"
        ) from exc

    return client


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

def exec_command(
    client: paramiko.SSHClient,
    cmd: str,
    timeout: float,
) -> tuple[int, str, str]:
    """Run a non-interactive command via /bin/bash login shell.

    Uses /bin/bash -l so that /etc/profile.d scripts are sourced,
    which initialises the HPC module system correctly over SSH.
    """
    bash_cmd = f'/bin/bash -l -c {shlex.quote(cmd)}'
    _, stdout, stderr = client.exec_command(bash_cmd, timeout=timeout)
    rc = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    LOGGER.warning("exit=%d stdout=%s stderr=%s", rc, out[:300], err[:300])
    return rc, out, err


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def _extract_uri_string(cell_value) -> str:
    """Extract a clean URI string from a KNIME URIDataCell value.

    The binary format of a URIDataCell is:
        [header control bytes] [URI bytes] [null byte] [EXT metadata bytes]
    """
    if isinstance(cell_value, dict):
        cell_value = list(cell_value.values())[0]

    if isinstance(cell_value, (bytes, bytearray)):
        cell_bytes = bytes(cell_value)
        scheme_marker = b"://"

        try:
            scheme_pos = cell_bytes.index(scheme_marker)
        except ValueError:
            return cell_bytes.decode("utf-8", errors="replace").strip()

        start = scheme_pos
        while start > 0 and chr(cell_bytes[start - 1]).isalpha():
            start -= 1

        after_start = cell_bytes[start:]
        null_pos = after_start.find(b"\x00")
        if null_pos != -1:
            uri_bytes = after_start[:null_pos]
        else:
            uri_bytes = after_start

        return uri_bytes.decode("utf-8", errors="replace").strip().rstrip("<>")

    else:
        raw = str(cell_value)
        if ";" in raw:
            raw = raw[:raw.index(";")]
        return raw.strip().rstrip("<>")


def path_from_row(row: pd.Series, path_col: str) -> tuple[str, str]:
    """Extract (subject_id, remote_path) from a List Files/Folders table row."""
    if path_col in row.index:
        cell = row[path_col]
    else:
        cell = row.iloc[0]
        LOGGER.warning("Column %r not found, using first column", path_col)

    raw = _extract_uri_string(cell)
    LOGGER.warning("path_from_row raw=%r", raw)

    if raw.startswith(("sftp://", "ssh://", "ftp://")):
        remote_path = urlparse(raw).path
    else:
        remote_path = raw.strip()

    remote_path = remote_path.strip()
    LOGGER.warning("path_from_row remote_path=%r", remote_path)

    subject_id = posixpath.basename(remote_path.rstrip("/"))
    return subject_id, remote_path


# ---------------------------------------------------------------------------
# CSV reading and normalisation
# ---------------------------------------------------------------------------

def read_remote_csv(sftp: paramiko.SFTPClient, remote_path: str) -> bytes:
    """Read a remote file via SFTP and return its raw bytes."""
    with sftp.open(remote_path, "rb") as fh:
        raw = fh.read()
    if not raw:
        raise IOError(f"File is empty: {remote_path}")
    return raw


def parse_stats_csv(raw: bytes, subject_id: str) -> pd.DataFrame:
    """Parse a FreeSurfer stats2table CSV and normalise its key column."""
    df = pd.read_csv(io.BytesIO(raw), sep=r"\t|,", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    if "Subject_ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Subject_ID"})

    df["Subject_ID"] = subject_id
    return df


def merge_subject_frames(frames: list) -> pd.DataFrame:
    """Outer-join a list of per-measure DataFrames on Subject_ID.

    Duplicate columns appearing in multiple stats files are deduplicated
    by keeping the first occurrence.
    """
    if not frames:
        return pd.DataFrame(columns=["Subject_ID"])

    merged = frames[0]
    for frame in frames[1:]:
        existing = set(merged.columns) - {"Subject_ID"}
        new_cols = [c for c in frame.columns if c == "Subject_ID" or c not in existing]
        frame = frame[new_cols]
        merged = pd.merge(merged, frame, on="Subject_ID", how="outer")

    return merged


# ---------------------------------------------------------------------------
# Hippocampus and Amygdala text file extraction
# ---------------------------------------------------------------------------

def read_hippo_amyg_features(
    sftp,
    subject_id: str,
    subject_path: str,
    include_hippocampus: bool,
    include_amygdala: bool,
) -> pd.DataFrame:
    """Read hippocampus and amygdala volumes from FreeSurfer mri/ text files.

    Source files:
        <subject>/mri/lh.hippoSfVolumes.txt   (hippocampal subfields, left)
        <subject>/mri/rh.hippoSfVolumes.txt   (hippocampal subfields, right)
        <subject>/mri/lh.amygNucVolumes.txt   (amygdala nuclei, left)
        <subject>/mri/rh.amygNucVolumes.txt   (amygdala nuclei, right)

    Each file is whitespace-delimited with two columns: Structure and Volume.
    Output is converted to wide format — one column per structure.
    """
    mri_dir = posixpath.join(subject_path, "mri")

    targets = []
    if include_hippocampus:
        targets += [
            ("lh_hippo", "left",  "lh.hippoSfVolumes.txt"),
            ("rh_hippo", "right", "rh.hippoSfVolumes.txt"),
        ]
    if include_amygdala:
        targets += [
            ("lh_amyg", "left",  "lh.amygNucVolumes.txt"),
            ("rh_amyg", "right", "rh.amygNucVolumes.txt"),
        ]

    frames = []
    for label, hemi, filename in targets:
        remote_path = posixpath.join(mri_dir, filename)
        try:
            with sftp.open(remote_path, "r") as fh:
                raw = fh.read()
            if not raw:
                LOGGER.warning("[%s] %s is empty, skipping", subject_id, filename)
                continue
        except IOError as exc:
            LOGGER.warning("[%s] Could not read %s: %s", subject_id, filename, exc)
            continue

        try:
            df = pd.read_csv(
                io.BytesIO(raw if isinstance(raw, bytes) else raw.encode()),
                delim_whitespace=True,
                header=None,
                names=["Structure", "Volume"],
            )
            df["Structure"] = df["Structure"].astype(str) + f"_{hemi}"
            df.insert(0, "Subject_ID", subject_id)

            df_wide = df.pivot(
                index="Subject_ID", columns="Structure", values="Volume"
            ).reset_index()
            df_wide.columns.name = None
            frames.append(df_wide)

        except Exception as exc:
            LOGGER.warning("[%s] Error parsing %s: %s", subject_id, filename, exc)
            continue

    if not frames:
        return pd.DataFrame(columns=["Subject_ID"])

    merged = frames[0]
    for frame in frames[1:]:
        existing = set(merged.columns) - {"Subject_ID"}
        new_cols = [c for c in frame.columns if c == "Subject_ID" or c not in existing]
        merged = pd.merge(merged, frame[new_cols], on="Subject_ID", how="outer")

    return merged
