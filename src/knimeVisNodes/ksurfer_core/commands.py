"""FreeSurfer command building and environment wrapping.

No KNIME or paramiko dependency — importable and testable standalone.
"""

import shlex
from dataclasses import dataclass

from .constants import MEASURE_KEYS


@dataclass
class Cmd:
    """One asegstats2table / aparcstats2table invocation for one subject."""
    label: str
    bare_cmd: str      # the FreeSurfer CLI call, WITHOUT env setup
    output_file: str   # filename only, written into subject's stats/ dir


def build_commands(subject_id: str, selected: set) -> list:
    """Return the list of FreeSurfer CLI commands for the selected measures."""
    cmds = []
    sid = shlex.quote(subject_id)

    if "aseg_volume" in selected:
        cmds.append(Cmd(
            label="aseg_volume",
            bare_cmd=f"asegstats2table --subjects {sid} --meas volume --tablefile aseg_volume.csv",
            output_file="aseg_volume.csv",
        ))

    aparc_measures = {
        "thickness":     "thickness",
        "thickness_std": "thicknessstd",
        "volume":        "volume",
        "meancurv":      "meancurv",
        "area":          "area",
    }
    for key, fs_meas in aparc_measures.items():
        if key not in selected:
            continue
        for hemi in ("lh", "rh"):
            out = f"{hemi}_{key}.csv"
            cmds.append(Cmd(
                label=f"{hemi}_{key}",
                bare_cmd=(
                    f"aparcstats2table --subjects {sid} --hemi {hemi} "
                    f"--meas {fs_meas} --tablefile {out}"
                ),
                output_file=out,
            ))

    # Note: hippocampus and amygdala are extracted from mri/ text files
    # directly over SFTP (see ssh.py: read_hippo_amyg_features).
    # They do NOT use asegstats2table commands.

    return cmds


def wrap_with_env(
    bare_cmd: str,
    subjects_dir: str,
    stats_dir: str,
    bashrc_path: str = "",
    freesurfer_module: str = "",
    freesurfer_home: str = "",
    python_module: str = "",
    venv_activate: str = "",
    module_init_script: str = "",
) -> str:
    """Wrap a bare FreeSurfer CLI command with the full HPC environment setup.

    Replicates the environment initialisation steps from a SLURM batch
    script. All parameters are optional — only non-empty values are
    included. Steps execute in the correct order:

        1. source <module_init_script>    initialise module system (if needed)
        2. source <bashrc_path>           load user environment
        3. module load <freesurfer_module> load FreeSurfer
        4. export FREESURFER_HOME=...     set FS home
        5. source .../SetUpFreeSurfer.sh  set up FS environment
        6. module load <python_module>    load Python module
        7. source <venv_activate>         activate Python venv
        8. export SUBJECTS_DIR=...        set subjects directory
        9. cd <stats_dir>                 move to subject stats dir
       10. <bare_cmd>                     run the FreeSurfer command

    Args:
        bare_cmd:           The raw asegstats2table / aparcstats2table call.
        subjects_dir:       Remote SUBJECTS_DIR path (required).
        stats_dir:          Remote subject stats/ directory (required).
        bashrc_path:        Full remote path to user .bashrc, e.g.
                            /home/clusterusers/jsmith/.bashrc
        freesurfer_module:  FreeSurfer module name, e.g.
                            freesurfer/7.4.1-gcc-12.1.0
        freesurfer_home:    Remote FREESURFER_HOME path, e.g.
                            /opt/share/spack/.../freesurfer-7.4.1-...
        python_module:      Python module name, e.g.
                            python/3.10.8-gcc-12.1.0-linux-ubuntu22.04-x86_64
        venv_activate:      Remote path to Python venv activate script, e.g.
                            /home/user/FS_env/bin/activate
        module_init_script: Remote path to module system init script, e.g.
                            /opt/share/modules/init/bash (advanced)
    """
    parts = []

    # Step 1: initialise module system if needed
    if module_init_script.strip():
        parts.append(f"source {shlex.quote(module_init_script.strip())}")

    # Step 2: source user .bashrc
    if bashrc_path.strip():
        parts.append(f"source {shlex.quote(bashrc_path.strip())}")

    # Step 3: load FreeSurfer module
    if freesurfer_module.strip():
        parts.append(
            f"module load {shlex.quote(freesurfer_module.strip())} 2>/dev/null"
        )

    # Steps 4-5: set FREESURFER_HOME and source SetUpFreeSurfer.sh
    if freesurfer_home.strip():
        fs = freesurfer_home.strip().rstrip("/")
        parts.append(f"export FREESURFER_HOME={shlex.quote(fs)}")
        parts.append(
            f"source {shlex.quote(fs + '/SetUpFreeSurfer.sh')} 2>/dev/null"
        )

    # Step 6: load Python module
    if python_module.strip():
        parts.append(
            f"module load {shlex.quote(python_module.strip())} 2>/dev/null"
        )

    # Step 7: activate Python venv
    if venv_activate.strip():
        parts.append(f"source {shlex.quote(venv_activate.strip())}")

    # Steps 8-9: set SUBJECTS_DIR and cd to stats dir
    parts.append(f"export SUBJECTS_DIR={shlex.quote(subjects_dir)}")
    parts.append(f"cd {shlex.quote(stats_dir)}")

    # Step 10: the actual FreeSurfer command
    parts.append(bare_cmd)

    return " && ".join(parts)
