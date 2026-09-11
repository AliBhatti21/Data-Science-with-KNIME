"""ksurfer_core — pure-Python extraction logic for the KSurfer KNIME nodes."""

from .constants import MEASURE_KEYS, HIPPO_AMYG_KEYS
from .commands import Cmd, build_commands, wrap_with_env
from .ssh import (
    open_client,
    exec_command,
    path_from_row,
    read_remote_csv,
    parse_stats_csv,
    merge_subject_frames,
    read_hippo_amyg_features,
)
from .connection_port import (
    KSurferConnectionSpec,
    KSurferConnectionPortObject,
    ksurfer_connection_port_type,
)

__all__ = [
    "MEASURE_KEYS",
    "HIPPO_AMYG_KEYS",
    "Cmd",
    "build_commands",
    "wrap_with_env",
    "open_client",
    "exec_command",
    "path_from_row",
    "read_remote_csv",
    "parse_stats_csv",
    "merge_subject_frames",
    "read_hippo_amyg_features",
    "KSurferConnectionSpec",
    "KSurferConnectionPortObject",
    "ksurfer_connection_port_type",
]
