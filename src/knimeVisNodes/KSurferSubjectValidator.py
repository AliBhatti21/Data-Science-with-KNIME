import sys
import os
import logging
import stat

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

from ksurfer_core.connection_port import (  
    KSurferConnectionSpec,
    KSurferConnectionPortObject,
    ksurfer_connection_port_type,
)

_DEFAULT_IGNORED = {"fsaverage", "fsaverage5", "fsaverage6", "lh", "rh"}


@knext.parameter_group(label="Subject Discovery")
class SubjectDiscoverySettings:
    subjects_dir = knext.StringParameter(
        label="SUBJECTS_DIR (remote path)",
        description=(
            "Absolute remote path to the FreeSurfer subjects root directory, "
            "e.g. `/data/3d_smt/ADNI1_FS_Output`."
        ),
        default_value="",
    )
    extra_ignored = knext.StringParameter(
        label="Additional folders to ignore (comma-separated)",
        description="Extra folder names to exclude beyond the built-in defaults.",
        default_value="",
    )
    only_complete = knext.BoolParameter(
        label="Only show complete subjects",
        description="Exclude subjects missing stats/, mri/, or surf/ directories.",
        default_value=False,
    )


@knext.node(
    name="Remote Subject Validator",
    node_type=knext.NodeType.SOURCE,
    icon_path="icons/valid.png",
    category=knimeVis_category,
    id="ksurfer-subject-validator",
)
@knext.input_port(
    name="KSurfer SSH Connection",
    description="Connection from KSurfer SSH Connector node.",
    port_type=ksurfer_connection_port_type,
)

@knext.output_table(
    name="Validated Subjects",
    description=(
        "One row per candidate FreeSurfer subject directory with validation flags."
    ),
)
class KSurferSubjectValidatorNode:
    """Scan SUBJECTS_DIR and validate FreeSurfer subject folders.

    Reuses credentials from the upstream KSurfer SSH Connector —
    no password required in this node.
    """

    discovery = SubjectDiscoverySettings()

    def configure(self, configure_context, connection_spec):
        if not self.discovery.subjects_dir:
            raise knext.InvalidParametersError("SUBJECTS_DIR must not be empty.")

        return knext.Schema(
                [knext.string(), knext.string(),
                 knext.bool_(), knext.bool_(), knext.bool_(),
                 knext.bool_(), knext.bool_(), knext.bool_()],
                ["Subject_ID", "remote_path",
                 "has_stats", "has_mri", "has_surf",
                 "looks_complete", "has_hippo_files", "has_amyg_files"],
            )
        

    def execute(self, exec_context, connection):
        import posixpath
        import pandas as pd

        _node_dir = os.path.dirname(os.path.realpath(__file__))
        if _node_dir not in sys.path:
            sys.path.insert(0, _node_dir)

        spec = connection.spec

        # Open fresh connection using stored credentials — no password needed
        try:
            client = spec.open_client()
        except RuntimeError as exc:
            raise RuntimeError(str(exc))

        ignored = _DEFAULT_IGNORED | {
            s.strip() for s in self.discovery.extra_ignored.split(",") if s.strip()
        }

        exec_context.set_progress(0.05, "Scanning SUBJECTS_DIR...")
        sftp = client.open_sftp()
        rows = []

        try:
            try:
                entries = sftp.listdir_attr(self.discovery.subjects_dir)
            except IOError as exc:
                raise RuntimeError(
                    f"Could not list SUBJECTS_DIR {self.discovery.subjects_dir!r}: {exc}"
                )

            n = max(len(entries), 1)
            for i, entry in enumerate(entries):
                name = entry.filename
                if name.startswith(".") or name in ignored:
                    continue
                if not stat.S_ISDIR(entry.st_mode):
                    continue

                subject_path = posixpath.join(self.discovery.subjects_dir, name)
                try:
                    sub_names = {e.filename for e in sftp.listdir_attr(subject_path)}
                except IOError:
                    continue

                has_stats = "stats" in sub_names
                has_mri   = "mri"   in sub_names
                has_surf  = "surf"  in sub_names
                looks_complete = has_stats and has_mri and has_surf

                has_hippo = has_amyg = False
                if has_mri:
                    mri_path = posixpath.join(subject_path, "mri")
                    try:
                        mri_files = {e.filename for e in sftp.listdir_attr(mri_path)}
                        has_hippo = (
                            "lh.hippoSfVolumes.txt" in mri_files and
                            "rh.hippoSfVolumes.txt" in mri_files
                        )
                        has_amyg = (
                            "lh.amygNucVolumes.txt" in mri_files and
                            "rh.amygNucVolumes.txt" in mri_files
                        )
                    except IOError:
                        pass

                if self.discovery.only_complete and not looks_complete:
                    continue

                rows.append({
                    "Subject_ID": name, "remote_path": subject_path,
                    "has_stats": has_stats, "has_mri": has_mri,
                    "has_surf": has_surf, "looks_complete": looks_complete,
                    "has_hippo_files": has_hippo, "has_amyg_files": has_amyg,
                })

                exec_context.set_progress(0.1 + 0.85 * (i / n),
                                          f"Validated {i+1}/{n}: {name}")
                if exec_context.is_canceled():
                    raise RuntimeError("Canceled by user.")

        finally:
            sftp.close()
            client.close()

        exec_context.set_progress(1.0, f"Found {len(rows)} subject(s).")

        if not rows:
            exec_context.set_warning(
                f"No valid subjects found under {self.discovery.subjects_dir!r}."
            )

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "Subject_ID", "remote_path", "has_stats", "has_mri", "has_surf",
            "looks_complete", "has_hippo_files", "has_amyg_files",
        ])

        return knext.Table.from_pandas(df)
