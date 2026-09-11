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
    name="Hippocampus Segmentation",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/brainSeg.ico",
    category=knimeVis_category,
    id="seg-Hippo"
)
@knext.input_table(
    name="Input Paths",
    description="Table with a string column of file paths to T1-weighted MRI volumes (.nii or .nii.gz)."
)
@knext.output_table(
    name="Segmented Slice",
    description="One 2D coronal overlay image per subject (left hippocampus = red, right = blue)."
)
@knext.output_table(
    name="Hippocampus Volumes",
    description="Left/right hippocampal volumes (mm³) and eTIV per subject."
)
class HippoDeepSegmentation:
    """
    Segments left and right hippocampus from raw T1-weighted MRI volumes using HippoDeep
    (https://github.com/bthyreau/hippodeep_pytorch, MIT License).

    Outputs:
      Port 1 — PNG overlay of the best coronal slice with hippocampus masks coloured
               left=red, right=blue, alpha proportional to segmentation confidence.
      Port 2 — Table with Hippocampus_Left_mm3, Hippocampus_Right_mm3, eTIV_mm3.
    """

    path_column = knext.ColumnParameter(
        label="MRI Path Column",
        description="Column containing file paths to T1-weighted 3D MRI volumes.",
        port_index=0,
        column_filter=kutil.is_string,
    )

    file_format = knext.StringParameter(
        label="File Format",
        description="Format of the MRI input files to process.",
        enum=[".nii", ".nii.gz"],
        default_value=".nii.gz",
    )

    def configure(self, configure_context, input_schema_1):
        path_columns = [c.name for c in input_schema_1 if kutil.is_string(c)]
        if not path_columns:
            raise ValueError("No string column found to use as MRI path column.")
        self.path_column = path_columns[-1]

        return (
            knext.Schema.from_columns([
                knext.Column(knext.string(), "Path"),
                knext.Column(knext.logical(Image.Image), "Segmented Slice"),
                knext.Column(knext.int64(), "Best_Slice_Index"),
            ]),
            knext.Schema.from_columns([
                #column names match exactly what execute() puts in the DataFrame.
                knext.Column(knext.string(), "Path"),
                knext.Column(knext.double(), "Hippocampus_Left_mm3"),
                knext.Column(knext.double(), "Hippocampus_Right_mm3"),
                knext.Column(knext.double(), "eTIV_mm3"),
            ]),
        )

    def execute(self, exec_context, input_table):
        
        # nibabel only breaks this node at run time, not the whole extension at startup.
        import nibabel as nib

        # Resolve the engine path and import.
        _node_dir = os.path.dirname(os.path.realpath(__file__))
        if _node_dir not in sys.path:
            sys.path.insert(0, _node_dir)

        try:
            from hippodeep_engine.hippodeep_core import segment_hippocampus
        except ImportError as e:
            raise RuntimeError(
                f"hippodeep_engine could not be imported from {_node_dir}. "
                f"Make sure hippodeep_engine/ (with hippodeep_core.py and torchparams/) "
                f"sits in the same folder as this node file.\nOriginal error: {e}"
            )

        input_df = input_table.to_pandas()

        if self.path_column not in input_df.columns:
            raise ValueError(f"Column '{self.path_column}' not found.")

        # use self.file_format to filter, not both formats hardcoded.
        filtered_df = input_df[
            input_df[self.path_column].str.endswith(self.file_format)
        ]

        if filtered_df.empty:
            raise ValueError(
                f"No files with format '{self.file_format}' found in column "
                f"'{self.path_column}'. Check the File Format parameter matches "
                f"your actual file extensions."
            )

        image_rows = []
        result_rows = []
        n = max(len(filtered_df.index), 1)

        for i, idx in enumerate(filtered_df.index):
            mri_path = str(filtered_df.at[idx, self.path_column])
            exec_context.set_progress(i / n, f"Segmenting {os.path.basename(mri_path)}")

            try:
                if not os.path.isfile(mri_path):
                    LOGGER.warning(f"File not found, skipping: {mri_path}")
                    image_rows.append((mri_path, None, -1))
                    result_rows.append([mri_path, None, None, None])
                    continue

                result = segment_hippocampus(mri_path)
                # Count how many hippocampus voxels appear in each slice, pick the slice with the most.
                
                combined = (result.mask_l > 0) | (result.mask_r > 0)
                ax = result.coronal_axis
                roi_per_slice = combined.sum(
                    axis=tuple(a for a in range(3) if a != ax))
                best_slice = (
                    int(np.argmax(roi_per_slice))
                    if roi_per_slice.max() > 0
                    else combined.shape[ax] // 2
                )

                t1_slice = np.take(
                    nib.load(mri_path).get_fdata().squeeze(),
                    best_slice, axis=ax
                )
                maskl_slice = np.take(result.mask_l, best_slice, axis=ax)
                maskr_slice = np.take(result.mask_r, best_slice, axis=ax)
                overlay = self._create_overlay(t1_slice, maskl_slice, maskr_slice)

                image_rows.append((mri_path, overlay, best_slice))
                result_rows.append([
                    mri_path,
                    result.hippo_l_mm3,
                    result.hippo_r_mm3,
                    result.etiv_mm3,
                ])

            except Exception as e:
                LOGGER.error(f"Error processing {mri_path}: {e}")
                image_rows.append((mri_path, None, -1))
                result_rows.append([mri_path, None, None, None])

        image_df = pd.DataFrame(
            image_rows,
            columns=["Path", "Segmented Slice", "Best_Slice_Index"]
        )
        result_df = pd.DataFrame(
            result_rows,
            columns=["Path", "Hippocampus_Left_mm3", "Hippocampus_Right_mm3", "eTIV_mm3"]
        )
        return knext.Table.from_pandas(image_df), knext.Table.from_pandas(result_df)

    def _create_overlay(self, t1_slice, maskl_slice, maskr_slice):
        t1 = t1_slice.astype(np.float32)
        t1 -= t1.min()
        if t1.max() > 0:
            t1 /= t1.max()
        rgb = np.stack([t1 * 255] * 3, axis=-1).astype(np.float32)

        al = (np.clip(maskl_slice / 255.0, 0, 1) * 0.6)[..., None]
        ar = (np.clip(maskr_slice / 255.0, 0, 1) * 0.6)[..., None]
        red  = np.array([255,  60,  60])
        blue = np.array([ 60, 120, 255])

        overlay = rgb * (1 - al) + red  * al
        overlay = overlay * (1 - ar) + blue * ar
        overlay = np.rot90(overlay.clip(0, 255).astype(np.uint8))
        return Image.fromarray(overlay)