import sys
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import knutils as kutil

torch.set_num_threads(1)  # Limits CPU threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = kutil.get_knimeVis_category ()

@knext.node(
    name="Automatic Segmentation (SAM)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/sam_icon.png",
    category=knimeVis_category,
    id="seg-SAM1"
)
@knext.input_table(name="Input Image", description="Input table containing the image objects.")
@knext.output_table(name="Segmented Image", description="Table containing SAM segmentated image results.")
@knext.output_table(name="Segmentation Results", description="Table containing segmentation results of SAM model such as bounding boxes, predicted IOU  and stability scores.")

class SAMSegmentation:
    """    
    SAM Automatic Image Segmentation
    This node performs automatic image segmentation using the Segment Anything Model (SAM) without prompts. It accepts a table containing image, a path of the model, whcih can be downloaded from the following link https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints ,to process the images, a model type(sam_checkpoints),  and outputs a table with the segmentation results along with bounding boxes (xywh), stability scores, and predicted IoU.

    Tuneable parameters have been defined in the advanced settings of the configuration node to improve the automatically generated segmentation masks. 

    """
    
    # Basic Parameters Section
    
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column containing the image objects.",
        port_index=0,
        column_filter=kutil.is_png,
    )
    Image_ID = knext.ColumnParameter(
        label= "Image ID",
        description= "Select the Image Path to identify image",
        port_index=0,
        column_filter = kutil.is_string

    )
    model_path = knext.LocalPathParameter(
        label="Model file path",
        description="Specify the path to the SAM model file (e.g., sam_vit_h_4b8939.pth).",
    )

    model_type = knext.StringParameter(
        label="Model type",
        description="SAM model type (e.g., 'vit_h', 'vit_l', 'vit_b')",
        default_value="vit_h",
        enum=["vit_h","vit_l","vit_b"]
    )
    device = knext.StringParameter(
        label="Device (GPU not available, falling back to CPU)",
        description="Select the device to run the SAM model.",
        default_value="CPU",
        enum=["CPU", "GPU"]
    )
    # Advanced Parameters Section

    points_per_side = knext.IntParameter(
        label="Points per side",
        description="""Number of points to sample along each image side for generating masks.
        Higher values detect more objects but increase computation time.""",
        default_value=32,
        min_value=4,
        max_value=128,
        is_advanced=True
    )
    
    pred_iou_thresh = knext.DoubleParameter(
        label="Predicted IoU threshold",
        description="""Filters masks based on predicted quality (0-1).
        Higher values retain only higher-quality masks.""",
        default_value=0.86,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True
    )
    
    stability_score_thresh = knext.DoubleParameter(
        label="Stability score threshold",
        description="""Filters masks based on stability across binarization thresholds (0-1).
        Higher values retain only more stable masks.""",
        default_value=0.92,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True
    )
    
    min_mask_region_area = knext.IntParameter(
        label="Minimum mask region area",
        description="""Removes small disconnected regions/holes (in pixels).
        Set to 0 to disable post-processing.""",
        default_value=100,
        min_value=0,
        max_value=10000,
        is_advanced=True
    )
    
    crop_n_layers = knext.IntParameter(
        label="Number of crop layers",
        description="""How many layers of crops to run segmentation on.
        0 = only original image, 1+ = additional crops.""",
        default_value=1,
        min_value=0,
        max_value=5,
        is_advanced=True
    )
    
    crop_n_points_downscale_factor = knext.IntParameter(
        label="Crop points downscale factor",
        description="""Reduces points_per_side in each crop layer by this factor.
        Higher values make crop layers faster but less detailed.""",
        default_value=2,
        min_value=1,
        max_value=4,
        is_advanced=True
    )
    
    crop_overlap_ratio = knext.DoubleParameter(
        label="Crop overlap ratio",
        description="""Overlap between crops (0-1).
        Higher values reduce missed objects at crop boundaries but increase computation.""",
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True
    )
    
    box_nms_thresh = knext.DoubleParameter(
        label="Box NMS threshold",
        description="""Non-maximum suppression threshold for overlapping masks (0-1).
        Higher values allow more overlapping masks.""",
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema):
        
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]
        if not image_columns:
            raise ValueError("No image columns available for image paths or objects.")
        self.image_column = image_columns[-1][0] #select the last PNG column from the table

        image_path = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_string(c)]
        if not image_path:
            raise ValueError("No image columns available for image paths or objects.")
        self.Image_ID = image_path[-1][0] #select the last PNG column from the table

        if not os.path.isfile(self.model_path):
            raise ValueError("Model file not found at the specified path.")
        
        # Extract model type from filename
        filename = os.path.basename(self.model_path)
        if not filename.startswith("sam_vit_"):
            raise ValueError(
                f"Invalid model filename format. Expected pattern: 'sam_vit_[h/l/b]_*.pth'"
                f"Got: {filename}"
            )
        
        # Get model type from filename (e.g., "h" from "sam_vit_h_4b8939.pth")
        file_model_type = filename.split("_")[2]  # splits into ['sam', 'vit', 'h', '4b8939.pth']
        
        # Validate match with selected type
        expected_prefix = f"vit_{file_model_type}"
        if self.model_type != expected_prefix:
            raise ValueError(
                f"Model type mismatch! File suggests '{expected_prefix}', "
                f"but selected '{self.model_type}'. "
                f"Filename: {filename}"
            )
        
        if self.Image_ID is None:
            self.Image_ID = input_schema_1[0]   #select the Path by default at position 0
        
        output_schema_1 = knext.Schema.from_columns([
            knext.Column(knext.string(), "Path"),
            knext.Column(knext.logical(Image.Image), "Segmented Image")
        ])
        
        output_schema_2 = knext.Schema.from_columns([
            knext.Column(knext.string(), "Path"),
            knext.Column(knext.int64(), "X_center"),
            knext.Column(knext.int64(), "Y_center"),
            knext.Column(knext.int64(), "Width"),
            knext.Column(knext.int64(), "Height"),
            knext.Column(knext.double(), "Predicted IoU"),
            knext.Column(knext.double(), "Stability Score")
        ])

        return output_schema_1, output_schema_2

    def execute(self, exec_context, input_table):
        input_df = input_table.to_pandas()
        
        # Validate columns
        if self.image_column not in input_df.columns:
            raise ValueError(f"Image column '{self.image_column}' not found")
        if "Path" not in input_df.columns:
            raise ValueError("Path column not found")

        # Load SAM model
        device = "cuda" if self.device == "GPU" and torch.cuda.is_available() else "cpu"
        if self.device == "GPU" and device == "cpu":
            exec_context.set_warning("GPU not available, using CPU")
        
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
            sam.to(device=device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
                box_nms_thresh=self.box_nms_thresh,
                crop_overlap_ratio=self.crop_overlap_ratio
            )
        except Exception as e:
            LOGGER.error(f"Failed to load SAM model: {e}")
            raise

        segmented_image_data = []
        segmentation_results = []

        for idx in input_df.index:
            row = {col: input_df.at[idx, col] for col in input_df.columns}
            try:
                path = str(row["Path"])
                image = row[self.image_column]
                
                if not isinstance(image, Image.Image):
                    LOGGER.warning(f"Skipping row {idx} - not a valid image")
                    continue

                # Convert image to numpy array
                image_np = np.array(image)
                if image_np.ndim == 3:
                    if image_np.shape[2] == 4:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                    else:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

                # Generate masks
                masks = mask_generator.generate(image_np)
                
                if masks:
                    vis_image = self._create_visualization(image_np, masks)
                    segmented_image_data.append((path, vis_image))
                    
                    for mask in masks:
                        x, y, w, h = mask['bbox']
                        segmentation_results.append([
                            path,
                            int(x),
                            int(y),
                            int(w),
                            int(h),
                            float(mask['predicted_iou']),
                            float(mask['stability_score'])
                        ])
                else:
                    LOGGER.warning(f"No masks found for {path}")
                    segmented_image_data.append((path, None))
                    
            except Exception as e:
                LOGGER.error(f"Error processing {path}: {str(e)}")
                segmented_image_data.append((path, None))

        # Create output DataFrames
        segmented_image_df = pd.DataFrame(segmented_image_data, 
                                        columns=["Path", "Segmented Image"])
        
        segmentation_results_df = pd.DataFrame(
            segmentation_results if segmentation_results else [],
            columns=["Path", "X_center", "Y_center", "Width", "Height", 
                    "Predicted IoU", "Stability Score"]
        )

        return (
            knext.Table.from_pandas(segmented_image_df),
            knext.Table.from_pandas(segmentation_results_df)
        )

# Create visualization image with mask overlays
    def _create_visualization(self, image_np, masks):
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Create a copy of the image to draw on
        vis_image = image_rgb.copy()
        
        # Sort masks by area (largest first)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        for mask in sorted_masks:
            # Create a random color for the mask
            color = np.random.randint(0, 256, 3).tolist()
            
            # Apply the mask with some transparency
            mask_area = mask['segmentation']
            vis_image[mask_area] = (vis_image[mask_area] * 0.7 + np.array(color) * 0.3).astype(np.uint8)
        
        return Image.fromarray(vis_image)
    