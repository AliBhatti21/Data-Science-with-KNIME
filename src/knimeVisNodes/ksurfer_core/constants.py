"""Shared constants for the KSurfer extraction pipeline."""

# CLI-based measures (asegstats2table / aparcstats2table)
MEASURE_KEYS = [
    "aseg_volume",
    "thickness",
    "thickness_std",
    "volume",
    "meancurv",
    "area",
]

# Text-file-based measures (read directly from mri/ via SFTP)
HIPPO_AMYG_KEYS = ["hippocampus", "amygdala"]
