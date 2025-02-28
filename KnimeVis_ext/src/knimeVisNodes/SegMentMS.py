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
    name="Segmentation",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/denoise.png",
    category=knimeVis_category,
    id="segment-image",
)
@knext.output_table(
    name="Bounding Boxes",
    description="Table containing bounding box coordinates",
)
@knext.output_table(
    name="Segmentation Masks",
    description="Table containing segmentation masks",
)
@knext.output_table(
    name="image Masks",
    description="Table containing segmentation masks",
)

@knext.input_table(
    name="Image Data",
    description="Table containing image",
)

class Segment:
    """
    Segment

    This class provides functionality to segment any image using a given model.
    It includes methods to load an image, apply a segmentation model, and return
    the segmented result.

    Attributes:
        model: The segmentation model used to process the image.

    Methods:
        load_image(image_path): Loads an image from the specified path.
        segment_image(): Applies the segmentation model to the loaded image.
        get_result(): Returns the segmented image.
    """

    # Define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply Segmentation.",
        port_index=0,
        column_filter=kutil.is_png
    )

    
    # Define the file path parameter
    model_path = knext.StringParameter(
        label="Model file path",
        description="Specify the path to the model file to be deplyed.",
    )

    
    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        
        # Filter string columns for image 
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c) ]

        if not image_columns:
            # If no string columns are found, raise an exception or return None
            raise ValueError("No string columns available for image paths.")
        
        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.info(f"Selected image column: {self.image_column}")
        print(knext.LogicalType.supported_value_types())

        if not os.path.isfile(self.model_path):
            LOGGER.error(f"Model file not found at: {self.model_path}")
            raise ValueError("Model file not found at the specified path.")

        
        # Return the updated schema
        output_schema_boxes = knext.Schema.from_columns([
            knext.Column(knext.double(), "img_id"),          # Image ID
            knext.Column(knext.list_(knext.double()), "boxs")  # List of box coordinates
        ])
        output_schema_masks = knext.Schema.from_columns([
            knext.Column(knext.double(), "img_id"),          # Image ID
            knext.Column(knext.list_(knext.double()), "masks") # List of mask values
        ])  

        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "ImageMasked")])

        # Return the output schemas
        return output_schema_boxes, output_schema_masks, output_schema

    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        # Load a pretrained YOLO model
        model = YOLO(self.model_path)
        df = input_table.to_pandas()
        
        # Initialize lists to collect results
        boxes_data = {"img_id": [], "boxs": []}
        masks_data = {"img_id": [], "masks": []}

        # for idx, row in df.iterrows():
        img = df[self.image_column].item()  # Get image path or data
        img_id = 0  # Use index as img_id (or replace with a column like row["id"])

        # Process image with YOLO model
        box, mask, img_res = self.process_image(model, img)

        if box is not None and len(box) > 0:
            # Convert tensor to numpy if needed
            for single_box in box:  # Iterate over each box (e.g., [x_min, y_min, x_max, y_max])
                print(single_box)
                boxes_data["img_id"].append(img_id)
                boxes_data["boxs"].append(single_box)  # Convert to list for schema
        
        if mask is not None and len(mask) > 0:
            # Convert tensor to numpy if needed
            for single_mask in mask:  # Iterate over each mask
                masks_data["img_id"].append(img_id)
                masks_data["masks"].append(single_mask.flatten().tolist())  # Convert to list for schema
        

        # Create output DataFrames
        boxes_df = pd.DataFrame(boxes_data)
        masks_df = pd.DataFrame(masks_data)
        df["ImageMasked"] = img_res

        output_schema_boxes = knext.Table.from_pandas(boxes_df)
        output_schema_masks = knext.Table.from_pandas(masks_df)
        output_schema = knext.Table.from_pandas(df)

        # Convert to KNIME tables
        return output_schema_boxes, output_schema_masks, output_schema


    
    
    def process_image(self, model, img):
        result = model.predict(img)  # Example: result could be a list or dict
       
        boxes = result[0].boxes.xywhn #Contains normalized [x_center, y_center, width, height] coordinates relative to the original image dimensions (values between 0-1)
        boxes = boxes.numpy()
        masks = result[0].masks.xyn #Contains normalized [x, y] coordinates relative to the original image dimensions (values between 0-1)
        img_res = Image.fromarray(result[0].plot()[:, :, ::-1])

        return boxes, masks, img_res
