import sys
import os
import logging
import posixpath
import io

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


@knext.parameter_group(label="Features to Extract")
class FeatureSettings:
    hippocampus = knext.BoolParameter(
        label="Hippocampal subfield volumes",
        description=(
            "Read lh.hippoSfVolumes.txt and rh.hippoSfVolumes.txt "
            "from <subject>/mri/ via SFTP."
        ),
        default_value=True,
    )
    amygdala = knext.BoolParameter(
        label="Amygdala nuclei volumes",
        description=(
            "Read lh.amygNucVolumes.txt and rh.amygNucVolumes.txt "
            "from <subject>/mri/ via SFTP."
        ),
        default_value=True,
    )


@knext.parameter_group(label="Execution")
class ExecSettings:
    max_workers = knext.IntParameter(
        "Max parallel subjects",
        "Concurrent SFTP sessions. Higher values are safe — no commands run.",
        4, min_value=1, max_value=32,
    )
    skip_on_error = knext.BoolParameter(
        "Skip subjects with missing files",
        "Skip subjects where hippo/amygdala files are missing.",
        True,
    )
    ignored_dirs = knext.StringParameter(
        "Subject folder names to ignore",
        "Comma-separated.",
        "fsaverage,fsaverage5,fsaverage6,lh,rh",
    )


@knext.node(
    name="Subcortical Measurements",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/table.png",
    category=knimeVis_category,
    id="ksurfer-hippo-amyg-reader",
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
    name="Hippo/Amygdala Features",
    description="One row per subject with hippocampal and/or amygdala volumes.",
)
class KSurferHippoAmygReaderNode:
    """Extract hippocampal subfield and amygdala nuclei volumes.

    Reads FreeSurfer text files directly from each subject's mri/ directory
    via SFTP. No FreeSurfer commands or environment setup required.
    Reuses credentials from the upstream KSurfer SSH Connector —
    no password required in this node.
    """

    features  = FeatureSettings()
    execution = ExecSettings()

    def configure(self, configure_context, connection_spec, subjects_schema):
        if not self.features.hippocampus and not self.features.amygdala:
            raise knext.InvalidParametersError(
                "Select at least one feature (Hippocampal subfields or Amygdala nuclei)."
            )
        if "Subject_ID" not in subjects_schema.column_names:
            raise knext.InvalidParametersError("Input table must contain 'Subject_ID' column.")
        if "remote_path" not in subjects_schema.column_names:
            raise knext.InvalidParametersError("Input table must contain 'remote_path' column.")
        return knext.Schema([knext.string()], ["Subject_ID"])

    def execute(self, exec_context, connection, subjects_table):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd

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

        include_hippo = self.features.hippocampus
        include_amyg  = self.features.amygdala
        n_total       = len(subjects)
        completed     = 0
        all_frames    = []
        had_errors    = False

        def get_targets():
            targets = []
            if include_hippo:
                targets += [("lh_hippo", "left",  "lh.hippoSfVolumes.txt"),
                            ("rh_hippo", "right", "rh.hippoSfVolumes.txt")]
            if include_amyg:
                targets += [("lh_amyg", "left",  "lh.amygNucVolumes.txt"),
                            ("rh_amyg", "right", "rh.amygNucVolumes.txt")]
            return targets

        def parse_txt(raw, subject_id, hemi):
            import pandas as pd
            df_long = pd.read_csv(
                io.BytesIO(raw), delim_whitespace=True,
                header=None, names=["Structure", "Volume"])
            df_long["Structure"] = df_long["Structure"].astype(str) + f"_{hemi}"
            df_long.insert(0, "Subject_ID", subject_id)
            df_wide = df_long.pivot(index="Subject_ID", columns="Structure",
                                    values="Volume").reset_index()
            df_wide.columns.name = None
            return df_wide

        def process_subject(subject_id, subject_path):
            errors = []
            try:
                client = spec.open_client()
            except RuntimeError as exc:
                return subject_id, None, [str(exc)]
            try:
                sftp    = client.open_sftp()
                mri_dir = posixpath.join(subject_path, "mri")
                frames  = []

                for label, hemi, filename in get_targets():
                    remote_path = posixpath.join(mri_dir, filename)
                    try:
                        with sftp.open(remote_path, "rb") as fh:
                            raw = fh.read()
                        if not raw:
                            raise IOError("File is empty")
                    except IOError as exc:
                        msg = f"[{subject_id}] {label}: {remote_path}: {exc}"
                        errors.append(msg)
                        LOGGER.warning(msg)
                        continue
                    try:
                        frames.append(parse_txt(raw, subject_id, hemi))
                        LOGGER.warning("[%s] %s: read OK", subject_id, label)
                    except Exception as exc:
                        errors.append(f"[{subject_id}] {label}: parse error: {exc}")

                if not frames:
                    return subject_id, None, errors

                merged = frames[0]
                for frame in frames[1:]:
                    existing = set(merged.columns) - {"Subject_ID"}
                    new_cols = [c for c in frame.columns if c == "Subject_ID" or c not in existing]
                    merged = merged.merge(frame[new_cols], on="Subject_ID", how="outer")
                return subject_id, merged, errors
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
            exec_context.set_warning("Some subjects had missing files. See KNIME console.")
        if not all_frames:
            exec_context.set_warning("No hippocampus/amygdala data found.")
            return knext.Table.from_pandas(pd.DataFrame(columns=["Subject_ID"]))

        import pandas as pd
        return knext.Table.from_pandas(pd.concat(all_frames, axis=0, ignore_index=True))
