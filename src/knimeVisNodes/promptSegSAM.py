import sys
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
import torch
import cv2, random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from utils import knutils as kutil

torch.set_num_threads(1)  # Limits CPU threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = kutil.get_knimeVis_category ()

@knext.node(
    name="Prompt Segmentation (SAM)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/sam_icon.png",
    category=knimeVis_category,
    id="seg-single-box"
)
@knext.input_table(
    name="Input Image", 
    description="Table containing the input image"
)
@knext.input_table(
    name="Bounding Boxes Table", 
    description="Table containing bounding box coordinates"
)
@knext.output_table(
    name="Binary Masks",
    description="Table containing the binary masks of the image"
)
@knext.output_table(
    name="Segmented Image",
    description="Table containing the segmented image with mask overlay"
)
class SamSegmentSingleObject:
    '''
    This KNIME extension node performs instance segmentation using Meta's Segment Anything Model (SAM). The node matches the image to its corresponding bounding boxes using the image ID, then applies SAM to segment the specified regions. Each execution supports a single prompt (one bounding box per image), converting the box coordinates into a detailed segmentation mask. Designed for interactive workflows, this node is ideal for refining manual annotations into AI-powered segmentations, ensuring accuracy in object detection and masking tasks.
    It requires two inputs:

    1) Image Input Port:
    It accepts an image along with a unique image identifier (ID). The image ID is used to match the image with its corresponding bounding boxes.

    2) Bounding Box Input Port:
    It takes a CSV file containing bounding box coordinates (generated via an Interactive Bounding Box Widget). Each bounding box is defined by its coordinates (e.g., x_min, y_min, x_max, y_max) and linked to an image ID.
    
    3) Functionality:
    The node applies SAM to the specified image using the provided bounding box as a single prompt. It generates precise segmentation masks for the objects within the bounding boxes. Further, It supports one prompt per image (single bounding box per execution).
    
    4) Use Case:
    Ideal for object segmentation tasks where manual bounding box annotations are available. It enables interactive AI-assisted segmentation by refining user-provided bounding boxes into high-quality masks.
    SAM Automatic Image Segmentation

    5) Confguration:
    It accepts a table containing image, a path of the model, whcih can be downloaded from the following link https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints ,to processes the images, a model type,  and outputs a table with the segmentation results along with bounding boxes.

    '''
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column containing the image objects",
        port_index=0,
        column_filter=kutil.is_png,
    )

    image_id = knext.IntParameter(
        label = "Image Index", 
        default_value=0,
        description="Index of the image to process (must match Image Index in bounding box table)"
    )
    
    iterate_all_boxes = knext.BoolParameter(
        "Iterate Over All Bounding Boxes", 
        default_value=False,
        description="Enable to segment using all bounding boxes for the selected image"
    )

    bbox_index = knext.IntParameter(
        label= "BBox Index", 
        default_value= -1,
        description="Bounding Box Index will be used to apply a single prompt on the selected image (used only if iteration is disabled)."
    )

    sam_checkpoint = knext.LocalPathParameter(
        label = "SAM Checkpoint",
        description = "Path to the .pth SAM model file"
    )

    model_type = knext.StringParameter(
        label= "Model Type", 
        default_value="vit_b",
        description="SAM model type: vit_h, vit_l, or vit_b",
        enum=["vit_h", "vit_l", "vit_b"]
    )

    device = knext.StringParameter(
        label="Choose Available Device (If CUDA not available, automatically falling back to CPU)",
        description="Select the device to run the SAM model.",
        default_value="cpu",
        enum=["CPU", "GPU"]
    )
    def configure(self, configure_context: knext.ConfigurationContext, 
                 image_schema: knext.Schema, bbox_schema: knext.Schema) -> knext.Schema:
        # Validate image column
        image_columns = [(c.name, c.ktype) for c in image_schema if kutil.is_png(c)]
        if not image_columns:
            raise ValueError("No image columns available for image paths or objects.")
        self.image_column = image_columns[-1][0] if image_columns else None
        
        # Validate bbox columns - updated to match your CSV
        required_bbox_cols = {"Image Index", "BBox Index", "X_min", "Y_min", "X_max", "Y_max"}
        if not required_bbox_cols.issubset(bbox_schema.column_names):
            missing = required_bbox_cols - set(bbox_schema.column_names)
            raise ValueError(f"Bounding box table missing required columns: {missing}")
        
        if not os.path.isfile(self.sam_checkpoint):
            raise ValueError("Model file not found at the specified path.")

        # First output: Individual binary masks
        output_schema_1 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "Image Index"),
            knext.Column(knext.int32(), "BBox Index"),  # Matches your CSV
            knext.Column(knext.logical(Image.Image), "Binary Mask"),
        ])

        # Second output: Combined segmentation
        output_schema_2 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "Image Index"),
            knext.Column(knext.logical(Image.Image), "Segmented Image")
        ])

        return output_schema_1, output_schema_2

    def execute(self, exec_context: knext.ExecutionContext, 
               image_input: knext.Table, bbox_input: knext.Table) -> knext.Table:
        try:
            # Convert inputs to pandas
            image_df = image_input.to_pandas()
            bbox_df = bbox_input.to_pandas()

            # Validate image selection
            if len(image_df) == 0:
                raise ValueError("Input images table is empty")
                
            # Get the selected image
            if "Image Index" in image_df.columns:
                img_match = image_df[image_df["Image Index"] == self.image_id]
                if len(img_match) == 0:
                    raise ValueError(f"No image found with Image Index {self.image_id}")
                selected_image = img_match.iloc[0][self.image_column]
            else:
                if self.image_id >= len(image_df):
                    raise ValueError(f"Image index {self.image_id} is out of range (0-{len(image_df)-1})")
                selected_image = image_df.iloc[self.image_id][self.image_column]

            # Convert image to numpy array
            if not isinstance(selected_image, Image.Image):
                raise ValueError("Selected image is not in PIL.Image format")
                
            image_np = np.array(selected_image)
            if image_np.ndim == 3 and image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif image_np.ndim == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

            # Find matching bounding boxes - using updated column names
            if self.iterate_all_boxes:
                bbox_matches = bbox_df[bbox_df["Image Index"] == self.image_id]
                if len(bbox_matches) == 0:
                    # Debug info
                    print(f"Debug - All Image Indices in CSV: {bbox_df['Image Index'].unique()}")
                    print(f"Debug - Searching for Image Index: {self.image_id}")
                    print(f"Debug - Sample rows: {bbox_df.head()}")
                    raise ValueError(f"Selected image {self.image_id} has no bounding boxes")
            else:
                bbox_matches = bbox_df[
                    (bbox_df["Image Index"] == self.image_id) & 
                    (bbox_df["BBox Index"] == self.bbox_index)  # Matches CSV
                ]
                if len(bbox_matches) == 0:
                    available_indices = bbox_df[bbox_df["Image Index"] == self.image_id]["BBox Index"].unique()
                    raise ValueError(f"Selected image {self.image_id} has no bbox with index {self.bbox_index}. Available indices: {available_indices}")

            # Load SAM model with selected device
            if self.device == "GPU":
                if not torch.cuda.is_available():
                    exec_context.set_warning("GPU not available, falling back to CPU")
                    device = "cpu"
                else:
                    device = "cuda"
            else:  # CPU case
                device = "cpu"
                
            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            predictor.set_image(image_np)

            # Initialize variables for both outputs
            individual_masks = []
            combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
            overlay = image_np.copy()

            # Process each bounding box
            for _, bbox_row in bbox_matches.iterrows():
                # Using your updated column names
                x1, y1, x2, y2 = map(int, [bbox_row["X_min"], bbox_row["Y_min"], bbox_row["X_max"], bbox_row["Y_max"]])
                box = np.array([[x1, y1, x2, y2]])

                masks, _, _ = predictor.predict(box=box, multimask_output=True)
                best_mask = masks[0]
                
                # Generate random color for this segmentation
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                
                # Store individual mask (Output Port 1)
                binary_mask = Image.fromarray((best_mask.astype(np.uint8)) * 255, mode='L')
                individual_masks.append({
                    "Image Index": self.image_id,
                    "BBox Index": bbox_row["BBox Index"],  # Matches your CSV
                    "Binary Mask": binary_mask,
                })
                
                # Add to combined mask (Output Port 2)
                combined_mask = np.logical_or(combined_mask, best_mask)
                overlay[best_mask] = color

            # Create combined output image (Output Port 2)
            result_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            combined_output = [{
                "Image Index": self.image_id,
                "Segmented Image": result_image
            }]

            # Create output DataFrames
            individual_masks_df = pd.DataFrame(individual_masks)
            combined_output_df = pd.DataFrame(combined_output)

            return knext.Table.from_pandas(individual_masks_df), knext.Table.from_pandas(combined_output_df)

        except Exception as e:
            exec_context.set_warning(f"Error during execution: {str(e)}")
            raise