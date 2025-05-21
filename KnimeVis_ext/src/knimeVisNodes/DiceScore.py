import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutills as kutil
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from PIL import Image 
import pandas as pd
import time
import torch
from torchvision import transforms
import os

# Logger setup
LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)

# Node definition
@knext.node(
    name="Dice Scores",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/objdet.png",
    category=knimeVis_category,
    id="box-eval",
)
@knext.output_table(
    name="DiceScore",
    description="Table containing Dice Scores",
)

@knext.input_table(
    name="Predicted Sementation mask",
    description="Table containing predicted Sementation mask",
)

@knext.input_table(
    name="Labels Sementation mask",
    description="Table containing Labeled Sementation mask",
)

class DiceScore:
    """
    Computes Dice Score, also known as the Sørensen–Dice coefficient, a key metric for evaluating segmentation models in deep learning. 
    It quantifies the overlap between predicted and ground truth segmentation masks.
    """

    # Define your parameter
    image_id = knext.ColumnParameter(
        label="Image ID",
        description="Select the column to use as ID.",
        port_index=0
    )

    # Define your parameter
    mask_column = knext.ColumnParameter(
        label="Mask colum",
        description="Select the column to use as Mask.",
        port_index=0
    )

    class DeviceOptions(knext.EnumParameterOptions):
        CPU = ("cpu", "Use CPU for inference")
        GPU = ("gpu", "Use GPU for inference if CUDA available.")

    device: str = knext.EnumParameter(
        label="Computation Device",
        description="Choose between CPU or GPU (CUDA) for model inference.",
        default_value=DeviceOptions.CPU.name,
        enum=DeviceOptions
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
            knext.Column(knext.double(), "DiceScore"),     # DiceScore
        ])
       
        # Return the output schemas
        return output_schema_boxes

    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table1: knext.Table,input_table2: knext.Table) -> knext.Table:
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Determine and log device
        device = "cuda" if self.device == "gpu" and torch.cuda.is_available() else "cpu"
        LOGGER.info(f"Using device: {device}")

        # Convert KNIME tables to pandas DataFrames
        df_pred = input_table1.to_pandas()
        df_gt = input_table2.to_pandas()

        # Ensure necessary columns exist
        if self.image_id not in df_pred.columns or self.image_id not in df_gt.columns:
            raise ValueError(f"Column '{self.image_id}' not found in input tables.")

         # Initialize results dictionary
        results = {"img_id": [], "DiceScore": []}

        for img_id in df_pred[self.image_id].unique():
            # Extract bounding boxes for this image
            pred_mask = df_pred[df_pred[self.image_id] == img_id]
            gt_mask = df_gt[df_gt[self.image_id] == img_id]

            if pred_mask.empty or gt_mask.empty:
                continue

            # Combine all predicted masks (logical OR)
            combined_pred = None
            for mask in pred_mask[self.mask_column]:
                tensor = transform(mask).squeeze(0) > 0.5  # [H, W] binary
                if combined_pred is None:
                    combined_pred = tensor
                else:
                    combined_pred = torch.logical_or(combined_pred, tensor)

            # Combine all ground-truth masks
            combined_gt = None
            for mask in gt_mask[self.mask_column]:
                tensor = transform(mask).squeeze(0) > 0.5
                if combined_gt is None:
                    combined_gt = tensor
                else:
                    combined_gt = torch.logical_or(combined_gt, tensor)

            # Convert to numpy for Dice computation
            pred_np = combined_pred.cpu().numpy().astype(np.bool_)
            gt_np = combined_gt.cpu().numpy().astype(np.bool_)

            dice = self.dice_score(pred_np, gt_np)

            results["img_id"].append(img_id)
            results["DiceScore"].append(dice)


        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        return knext.Table.from_pandas(results_df)

    @staticmethod
    def dice_score(mask_pred, mask_true, smooth=1e-6):
        mask_pred = mask_pred.astype(np.bool_)
        mask_true = mask_true.astype(np.bool_)

        intersection = np.logical_and(mask_pred, mask_true).sum()
        total = mask_pred.sum() + mask_true.sum()

        dice = (2. * intersection + smooth) / (total + smooth)
        return dice

