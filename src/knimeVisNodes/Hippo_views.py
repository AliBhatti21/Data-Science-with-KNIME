import sys
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext

from utils import knutils as kutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LOGGER = logging.getLogger(__name__)

knimeVis_category = kutil.get_knimeVis_category()


@knext.node(
    name="Hippocampus View",
    node_type=knext.NodeType.MANIPULATOR ,
    icon_path="icons/brain.png",
    category=knimeVis_category,
    id="seg-Hippo3View"
)
@knext.input_table(
    name="Input Paths",
    description="Table with a string column of file paths to T1-weighted MRI volumes (.nii or .nii.gz)."
)
@knext.output_table(
    name="Three-View Slices",
    description=(
        "One row per subject with three PNG overlay images: "
        "Axial (top-down), Coronal (front-back), and Sagittal (left-right). "
        "Each slice is chosen at the position where the combined hippocampus "
        "mask has the largest cross-sectional area in that plane. "
        "Left hippocampus = red, right hippocampus = blue."
    )
)
class HippoDeepViewer:
    """
    Hippocampus View Node

    Runs HippoDeep segmentation (same engine and bundled weights as the
    Hippocampus Segmentation node) and renders three anatomical views of the
    result — axial, coronal, and sagittal — each sliced at the position that
    maximises hippocampus visibility in that plane.

    Useful for visual QC, side-by-side CN vs AD comparison, and presentations.
    No FreeSurfer, no subprocess, no external tool required.
    """

    path_column = knext.ColumnParameter(
        label="MRI Path Column",
        description="Column containing file paths to T1-weighted 3D MRI volumes.",
        port_index=0,
        column_filter=kutil.is_string,
    )

    file_format = knext.StringParameter(
        label="File Format",
        description="Format of the MRI input files.",
        enum=[".nii", ".nii.gz"],
        default_value=".nii.gz",
    )

    overlay_alpha = knext.DoubleParameter(
        label="Mask overlay opacity",
        description="Opacity of the hippocampus mask overlay (0 = invisible, 1 = opaque).",
        default_value=0.6,
        min_value=0.1,
        max_value=1.0,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema_1):
        path_columns = [c.name for c in input_schema_1 if kutil.is_string(c)]
        if not path_columns:
            raise ValueError("No string column found to use as MRI path column.")
        self.path_column = path_columns[-1]

        return knext.Schema.from_columns([
            knext.Column(knext.string(),              "Path"),
            knext.Column(knext.logical(Image.Image),  "Axial_Slice"),
            knext.Column(knext.logical(Image.Image),  "Coronal_Slice"),
            knext.Column(knext.logical(Image.Image),  "Sagittal_Slice"),
            knext.Column(knext.int64(),               "Axial_Slice_Index"),
            knext.Column(knext.int64(),               "Coronal_Slice_Index"),
            knext.Column(knext.int64(),               "Sagittal_Slice_Index"),
        ])

    def execute(self, exec_context, input_table):
        import nibabel as nib
        # Resolve and import the bundled engine in-process
        _node_dir = os.path.dirname(os.path.realpath(__file__))
        if _node_dir not in sys.path:
            sys.path.insert(0, _node_dir)
        try:
            from hippodeep_engine.hippodeep_core import segment_hippocampus
        except ImportError as e:
            raise RuntimeError(
                f"hippodeep_engine could not be imported from {_node_dir}.\n"
                f"Ensure hippodeep_engine/ (with hippodeep_core.py and torchparams/) "
                f"sits in the same folder as this node file.\nError: {e}"
            )

        input_df = input_table.to_pandas()
        if self.path_column not in input_df.columns:
            raise ValueError(f"Column '{self.path_column}' not found.")

        filtered_df = input_df[
            input_df[self.path_column].str.endswith(self.file_format)
        ]
        if filtered_df.empty:
            raise ValueError(
                f"No files with format '{self.file_format}' found in "
                f"'{self.path_column}'. Check the File Format parameter."
            )

        rows = []
        n = max(len(filtered_df.index), 1)

        for i, idx in enumerate(filtered_df.index):
            mri_path = str(filtered_df.at[idx, self.path_column])
            exec_context.set_progress(
                i / n, f"Segmenting {os.path.basename(mri_path)}")

            try:
                if not os.path.isfile(mri_path):
                    LOGGER.warning(f"File not found, skipping: {mri_path}")
                    rows.append(self._empty_row(mri_path))
                    continue

                # Run HippoDeep — same call as the Segmentation node
                result = segment_hippocampus(mri_path)

                # Load T1 for display (squeeze any trailing singleton dims)
                t1 = nib.load(mri_path).get_fdata().squeeze()

                # Combined mask: any voxel belonging to left or right hippocampus
                combined = (result.mask_l > 0) | (result.mask_r > 0)

                # Derive the three anatomical axes from the NIfTI affine so the
                # node works correctly for any scanner orientation, not just RAS.
                #   World X (left-right)       → sagittal plane
                #   World Y (anterior-posterior)→ coronal plane
                #   World Z (inferior-superior) → axial plane
                vox_to_world = result.affine[:3, :3]
                sag_axis     = int(np.argmax(np.abs(vox_to_world[0, :])))
                cor_axis     = int(np.argmax(np.abs(vox_to_world[1, :])))
                ax_axis      = int(np.argmax(np.abs(vox_to_world[2, :])))

                ax_idx  = self._best_slice(combined, ax_axis)
                cor_idx = self._best_slice(combined, cor_axis)
                sag_idx = self._best_slice(combined, sag_axis)

                ax_img  = self._render_slice(t1, result.mask_l, result.mask_r,
                                              ax_idx,  ax_axis,  rotate=True)
                cor_img = self._render_slice(t1, result.mask_l, result.mask_r,
                                              cor_idx, cor_axis, rotate=True)
                sag_img = self._render_slice(t1, result.mask_l, result.mask_r,
                                              sag_idx, sag_axis, rotate=True)

                rows.append([
                    mri_path,
                    ax_img,  cor_img,  sag_img,
                    ax_idx,  cor_idx,  sag_idx,
                ])

            except Exception as e:
                LOGGER.error(f"Error processing {mri_path}: {e}")
                rows.append(self._empty_row(mri_path))

        out_df = pd.DataFrame(rows, columns=[
            "Path",
            "Axial_Slice", "Coronal_Slice", "Sagittal_Slice",
            "Axial_Slice_Index", "Coronal_Slice_Index", "Sagittal_Slice_Index",
        ])
        return knext.Table.from_pandas(out_df)

    
    # Helpers
    # -------

    def _best_slice(self, combined_mask: np.ndarray, axis: int) -> int:
        """
        Find the slice index along `axis` where the combined hippocampus
        mask has the largest cross-sectional area (most nonzero voxels).
        Falls back to the middle slice if the mask is empty.
        """
        other_axes = tuple(a for a in range(3) if a != axis)
        roi_per_slice = combined_mask.sum(axis=other_axes)
        if roi_per_slice.max() > 0:
            return int(np.argmax(roi_per_slice))
        return combined_mask.shape[axis] // 2

    def _render_slice(
        self,
        t1: np.ndarray,
        mask_l: np.ndarray,
        mask_r: np.ndarray,
        slice_idx: int,
        axis: int,
        rotate: bool = True,
    ) -> Image.Image:
        """
        Extract one 2D slice from the T1 and both masks, then composite
        the masks as a semi-transparent coloured overlay on the grayscale MRI.
        Left hippocampus = red, right hippocampus = blue.
        """
        t1_slice  = np.take(t1,     slice_idx, axis=axis).astype(np.float32)
        ml_slice  = np.take(mask_l, slice_idx, axis=axis).astype(np.float32)
        mr_slice  = np.take(mask_r, slice_idx, axis=axis).astype(np.float32)

        # Normalise T1 to 0-255 grayscale
        t1_slice -= t1_slice.min()
        if t1_slice.max() > 0:
            t1_slice /= t1_slice.max()
        rgb = np.stack([t1_slice * 255] * 3, axis=-1).astype(np.float32)

        # Blend mask confidence (0-255 from HippoDeep) with user opacity
        al = (np.clip(ml_slice / 255.0, 0, 1) * self.overlay_alpha)[..., None]
        ar = (np.clip(mr_slice / 255.0, 0, 1) * self.overlay_alpha)[..., None]
        red  = np.array([255,  60,  60], dtype=np.float32)
        blue = np.array([ 60, 120, 255], dtype=np.float32)

        overlay = rgb  * (1 - al) + red  * al
        overlay = overlay * (1 - ar) + blue * ar
        overlay = overlay.clip(0, 255).astype(np.uint8)

        if rotate:
            overlay = np.rot90(overlay)

        return Image.fromarray(overlay)

    def _empty_row(self, path: str) -> list:
        """Placeholder row for a subject that failed or was not found."""
        return [path, None, None, None, -1, -1, -1]