import sys
import os
import logging
import posixpath
import knime.extension as knext
import torch
torch.set_num_threads(1)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import knutils as kutil

LOGGER = logging.getLogger(__name__)
knimeVis_category = kutil.get_knimeVis_category()

_node_dir = os.path.dirname(os.path.realpath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

from ksurfer_core.connection_port import (  # noqa: E402
    KSurferConnectionSpec,
    KSurferConnectionPortObject,
    ksurfer_connection_port_type,
)

_MEASURE_KEYS = ["aseg_volume", "thickness", "thickness_std",
                 "volume", "meancurv", "area"]


@knext.parameter_group(label="HPC Environment Setup")
class FsEnvSettings:
    subjects_dir = knext.StringParameter(
        label="SUBJECTS_DIR — FreeSurfer subjects root (required)",
        description="Absolute remote path to the FreeSurfer subjects root directory.",
        default_value="",
    )
    bashrc_path = knext.StringParameter(
        label="User .bashrc path",
        description="Full remote path to the user .bashrc, e.g. `/home/user/.bashrc`.",
        default_value="",
    )
    freesurfer_module = knext.StringParameter(
        label="FreeSurfer module name",
        description="Module name to load, e.g. `freesurfer/7.4.1-gcc-12.1.0`.",
        default_value="",
    )
    freesurfer_home = knext.StringParameter(
        label="FREESURFER_HOME — FreeSurfer installation path",
        description="Absolute remote path to FreeSurfer installation root.",
        default_value="",
    )
    python_module = knext.StringParameter(
        label="Python module name",
        description="Python module to load, e.g. `python/3.10.8-gcc-12.1.0-...`.",
        default_value="",
    )
    venv_activate = knext.StringParameter(
        label="Python venv activate script",
        description="Full remote path to the venv activate script.",
        default_value="",
    )
    module_init_script = knext.StringParameter(
        label="Module system init script (advanced)",
        description="Remote path to module system init, e.g. `/opt/share/modules/init/bash`.",
        default_value="", is_advanced=True,
    )


@knext.parameter_group(label="Measures to Extract")
class MeasureSettings:
    aseg_volume   = knext.BoolParameter("Subcortical volumes (aseg)",  "asegstats2table --meas volume",                True)
    thickness     = knext.BoolParameter("Cortical thickness",          "aparcstats2table --meas thickness, lh+rh",    True)
    thickness_std = knext.BoolParameter("Thickness std. dev.",         "aparcstats2table --meas thicknessstd, lh+rh", False)
    volume        = knext.BoolParameter("Cortical volume",             "aparcstats2table --meas volume, lh+rh",       True)
    meancurv      = knext.BoolParameter("Mean curvature",              "aparcstats2table --meas meancurv, lh+rh",     True)
    area          = knext.BoolParameter("Surface area",                "aparcstats2table --meas area, lh+rh",         True)

    def selected(self):
        return {k for k in _MEASURE_KEYS if getattr(self, k)}


@knext.parameter_group(label="Execution")
class ExecSettings:
    max_workers = knext.IntParameter("Max parallel subjects", "Concurrent SSH sessions.", 2, min_value=1, max_value=32)
    cmd_timeout = knext.IntParameter("Per-command timeout (s)", "Seconds per command.", 300, min_value=30, is_advanced=True)
    skip_on_error = knext.BoolParameter("Skip failed subjects/measures", "Log failures as warnings instead of aborting.", True)
    ignored_dirs = knext.StringParameter("Subject folder names to ignore", "Comma-separated.", "fsaverage,fsaverage5,fsaverage6,lh,rh")


@knext.node(
    name="Morphometric Measurements",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/table.png",
    category=knimeVis_category,
    id="ksurfer-feature-extractor",
)
@knext.input_port(
    name="KSurfer SSH Connection",
    description="Connection port from KSurfer SSH Connector node.",
    port_type=ksurfer_connection_port_type,
)
@knext.input_table(
    name="Subject Paths",
    description="Validated subject table from KSurfer Subject Validator (after Row Filter).",
)
@knext.output_table(
    name="Extracted Features",
    description="One row per subject with all selected FreeSurfer CLI-based measures.",
)
class KSurferFeatureExtractorNode:
    """Extract FreeSurfer morphometric features from HPC output over SSH.

    Reuses credentials from the upstream KSurfer SSH Connector
    """

    fs_env    = FsEnvSettings()
    measures  = MeasureSettings()
    execution = ExecSettings()

    def configure(self, configure_context, connection_spec, subjects_schema):
        if not self.fs_env.subjects_dir:
            raise knext.InvalidParametersError("SUBJECTS_DIR must not be empty.")
        if not self.measures.selected():
            raise knext.InvalidParametersError("Select at least one measure.")
        if "Subject_ID" not in subjects_schema.column_names:
            raise knext.InvalidParametersError("Input table must contain 'Subject_ID' column.")
        if "remote_path" not in subjects_schema.column_names:
            raise knext.InvalidParametersError("Input table must contain 'remote_path' column.")
        return knext.Schema([knext.string()], ["Subject_ID"])

    def execute(self, exec_context, connection, subjects_table):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd

        _node_dir = os.path.dirname(os.path.realpath(__file__))
        if _node_dir not in sys.path:
            sys.path.insert(0, _node_dir)

        from ksurfer_core import (
            build_commands, exec_command, merge_subject_frames,
            parse_stats_csv, read_remote_csv, wrap_with_env,
        )

        spec = connection.spec
        df   = subjects_table.to_pandas()

        if df.empty:
            exec_context.set_warning("Input table is empty.")
            return knext.Table.from_pandas(pd.DataFrame(columns=["Subject_ID"]))

        ignored = {s.strip() for s in self.execution.ignored_dirs.split(",") if s.strip()}
        subjects = [
            (str(row["Subject_ID"]), str(row["remote_path"]))
            for _, row in df.iterrows()
            if str(row["Subject_ID"]) not in ignored
        ]

        if not subjects:
            exec_context.set_warning("No subjects remain after filtering.")
            return knext.Table.from_pandas(pd.DataFrame(columns=["Subject_ID"]))

        selected   = self.measures.selected()
        n_total    = len(subjects)
        completed  = 0
        all_frames = []
        had_errors = False

        def process_subject(subject_id, subject_path):
            errors = []
            # Each call opens a fresh connection using stored credentials
            try:
                client = spec.open_client()
            except RuntimeError as exc:
                return subject_id, None, [str(exc)]
            try:
                sftp      = client.open_sftp()
                stats_dir = posixpath.join(subject_path, "stats")
                LOGGER.warning("[%s] stats_dir=%r", subject_id, stats_dir)
                try:
                    sftp.stat(stats_dir)
                except IOError:
                    sftp.mkdir(stats_dir)

                subject_frames = []
                for cmd in build_commands(subject_id, selected):
                    full_cmd = wrap_with_env(
                        bare_cmd=cmd.bare_cmd,
                        subjects_dir=self.fs_env.subjects_dir,
                        stats_dir=stats_dir,
                        bashrc_path=self.fs_env.bashrc_path,
                        freesurfer_module=self.fs_env.freesurfer_module,
                        freesurfer_home=self.fs_env.freesurfer_home,
                        python_module=self.fs_env.python_module,
                        venv_activate=self.fs_env.venv_activate,
                        module_init_script=self.fs_env.module_init_script,
                    )
                    LOGGER.warning("[%s] running: %s", subject_id, cmd.label)
                    rc, _out, err = exec_command(client, full_cmd, self.execution.cmd_timeout)
                    if rc != 0:
                        msg = f"[{subject_id}] {cmd.label} exited {rc}: {err.strip()[:250]}"
                        errors.append(msg)
                        LOGGER.warning(msg)
                        continue
                    remote_csv = posixpath.join(stats_dir, cmd.output_file)
                    try:
                        raw = read_remote_csv(sftp, remote_csv)
                    except IOError as exc:
                        errors.append(f"[{subject_id}] {cmd.label}: SFTP read failed: {exc}")
                        continue
                    try:
                        frame = parse_stats_csv(raw, subject_id)
                    except Exception as exc:
                        errors.append(f"[{subject_id}] {cmd.label}: parse error: {exc}")
                        continue
                    subject_frames.append(frame)

                if not subject_frames:
                    return subject_id, None, errors
                return subject_id, merge_subject_frames(subject_frames), errors
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=self.execution.max_workers) as pool:
            futures = {pool.submit(process_subject, sid, rpath): sid for sid, rpath in subjects}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    subject_id, frame, errors = future.result()
                except Exception as exc:
                    errors, frame, subject_id = [f"[{sid}] unexpected error: {exc}"], None, sid
                    LOGGER.error(errors[0])
                for msg in errors:
                    had_errors = True
                    if not self.execution.skip_on_error:
                        raise RuntimeError(f"Aborting: {msg}")
                if frame is not None:
                    all_frames.append(frame)
                completed += 1
                exec_context.set_progress(completed / n_total, f"Processed {completed}/{n_total}: {subject_id}")
                if exec_context.is_canceled():
                    raise RuntimeError("Canceled by user.")

        if had_errors:
            exec_context.set_warning("Some subjects/measures failed. See KNIME console.")
        if not all_frames:
            exec_context.set_warning("No data extracted.")
            return knext.Table.from_pandas(pd.DataFrame(columns=["Subject_ID"]))

        return knext.Table.from_pandas(pd.concat(all_frames, axis=0, ignore_index=True))
