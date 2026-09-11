import knime.extension as knext
import numpy as np
import logging
from utils import knutils as kutil
import pandas as pd

# Logger setup
LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = kutil.get_knimeVis_category()

# Node definition
@knext.node(
    name="Intersection over Union",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/objdet.png",
    category=knimeVis_category,
    id="box-eval",
)
@knext.output_table(
    name="IoU",
    description="Table containing Intersection over Union",
)

@knext.input_table(
    name="Predict Bounding Box Data",
    description="Table containing Bounding Box data in form xywhn",
)

@knext.input_table(
    name="Labels Bounding Box Data",
    description="Table containing Bounding Box data in form xywhn",
)

class IoU:
    """
    Computes Intersection over Union (IoU) for bounding boxes.
    """

    # Define your parameter
    image_id = knext.ColumnParameter(
        label="Image ID",
        description="Select the column to use as ID.",
        port_index=0
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
        input_schema_2: knext.Schema
    ):
        

        if self.image_id is None:
            self.image_id = input_schema_1[0].name  # Default to first column
       
        LOGGER.info(f"Selected image ID column: {self.image_id}")

        # Define output schema
        output_schema_boxes = knext.Schema.from_columns([
            knext.Column(knext.string(), "img_id"),  # Image ID
            knext.Column(knext.double(), "IoU"),     # IoU score
        ])
       
        # Return the output schemas
        return output_schema_boxes

    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table1: knext.Table,input_table2: knext.Table) -> knext.Table:

        # Convert KNIME tables to pandas DataFrames
        df_pred = input_table1.to_pandas()
        df_gt = input_table2.to_pandas()

        # Ensure necessary columns exist
        if self.image_id not in df_pred.columns or self.image_id not in df_gt.columns:
            raise ValueError(f"Column '{self.image_id}' not found in input tables.")

         # Initialize results dictionary
        results = {"img_id": [], "IoU": []}

        for img_id in df_pred[self.image_id].unique():
            # Extract bounding boxes for this image
            pred_boxes = df_pred[df_pred[self.image_id] == img_id][["x_center", "y_center", "width", "height"]].values
            gt_boxes = df_gt[df_gt[self.image_id] == img_id][["x_center", "y_center", "width", "height"]].values

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_scores = self.compute_iou(pred_boxes, gt_boxes)

                for iou in iou_scores:
                    results["img_id"].append(img_id)
                    results["IoU"].append(iou)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        return knext.Table.from_pandas(results_df)

    @staticmethod
    def compute_iou(pred_boxes, gt_boxes):
        """
        Compute IoU between predicted and ground truth bounding boxes.
        """

        # Convert [x, y, w, h] to [x1, y1, x2, y2] for both pred and gt
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
        gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
        gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
        gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

        # Compute intersection
        inter_x1 = np.maximum(pred_x1, gt_x1)
        inter_y1 = np.maximum(pred_y1, gt_y1)
        inter_x2 = np.minimum(pred_x2, gt_x2)
        inter_y2 = np.minimum(pred_y2, gt_y2)

        # Calculate intersection area
        inter_w = np.maximum(inter_x2 - inter_x1, 0)
        inter_h = np.maximum(inter_y2 - inter_y1, 0)
        inter_area = inter_w * inter_h

        # Compute union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area

        # Compute IoU
        iou = inter_area / (union_area + 1e-6)  # Small epsilon to avoid division by zero
        return iou

