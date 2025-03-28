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
    name="YOLO",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/objdet.png",
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
        (model can be found here: https://docs.ultralytics.com/tasks/segment/)

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

    image_id = knext.ColumnParameter(
        label="Image ID",
        description="Select the column to use as ID.",
        port_index=0
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

        if self.image_id is None:
            self.image_id = input_schema_1[0]
       
        # Log selected column
        LOGGER.info(f"Selected image column: {self.image_column}")
        print(knext.LogicalType.supported_value_types())

        if not os.path.isfile(self.model_path):
            LOGGER.error(f"Model file not found at: {self.model_path}")
            raise ValueError("Model file not found at the specified path.")

        
        # Return the updated schema
        output_schema_boxes = knext.Schema.from_columns([
            knext.Column(knext.string(), "img_id"),  
            knext.Column(knext.string(), "class"),  
            knext.Column(knext.double(), "confidence"),        
            knext.Column(knext.double(), "x_center"),  
            knext.Column(knext.double(), "y_center"),  
            knext.Column(knext.double(), "width"),  
            knext.Column(knext.double(), "height"),  
        ])
        output_schema_masks = knext.Schema.from_columns([
            knext.Column(knext.string(), "img_id"),
            knext.Column(knext.string(), "class"),  
            knext.Column(knext.double(), "confidence"), 
            knext.Column(Image.Image, "masks") # List of mask values
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
        boxes_data = {"img_id": [],"class":[],"confidence":[], "x_center": [], "y_center": [], "width": [], "height": []}
        masks_data = {"img_id": [],"class":[],"confidence":[], "masks": []}

        for img_id, img in zip(df[self.image_id],df[self.image_column]):
    

            # Process image with YOLO model
            box, mask, img_res, conf,classid = self.process_image(model, img)

            if box is not None and len(box) > 0:
                # Convert tensor to numpy if needed
                for i,single_box in enumerate(box):  # Iterate over each box (e.g., [x_min, y_min, x_max, y_max])
                    boxes_data["img_id"].append(img_id)
                    boxes_data["confidence"].append(conf[i])
                    boxes_data["class"].append(classid[i])
                    boxes_data["x_center"].append(single_box[0])  # Convert to list for schema
                    boxes_data["y_center"].append(single_box[1])
                    boxes_data["width"].append(single_box[2])
                    boxes_data["height"].append(single_box[3])
            
            if mask is not None and len(mask) > 0:
                # Convert tensor to numpy if needed
                for i,single_mask in enumerate(mask):  # Iterate over each mask
                    masks_data["img_id"].append(img_id)
                    masks_data["confidence"].append(conf[i])
                    masks_data["class"].append(classid[i])
                    masks_data["masks"].append(Image.fromarray((single_mask), mode="L"))  # Convert to list for schema
           
        # Create output DataFrames
        boxes_df = pd.DataFrame(boxes_data)
        masks_df = pd.DataFrame(masks_data)
        # df["ImageMasked"] = img_res

        output_schema_boxes = knext.Table.from_pandas(boxes_df)
        output_schema_masks = knext.Table.from_pandas(masks_df)
        output_schema = knext.Table.from_pandas(df)

        # Convert to KNIME tables
        return output_schema_boxes, output_schema_masks, output_schema


    
    
    def process_image(self, model, img):
        result = model.predict(img)  # Example: result could be a list or dict
        try:
            boxes = result[0].boxes.xywhn #Contains normalized [x_center, y_center, width, height] coordinates relative to the original image dimensions (values between 0-1)
            boxes = boxes.numpy()
        except:
            LOGGER.info(f"Bounding boxes not available")
            boxes = np.zeros((1, 4))  

        try:
            masks = result[0].masks.data 
            masks = masks.cpu().numpy()
            masks = (masks>0).astype(np.uinit8)*255
        except Exception as e:
            LOGGER.info(f"Masks not available: {str(e)}")
            masks = None  # Empty array for masks (no coordinates)  

        class_name = [result[0].names[int(cls)] for cls in result[0].boxes.cls]
        confidence = result.boxes.conf.cpu().numpy()
      
        img_res = Image.fromarray(result[0].plot()[:, :, ::-1])

        return boxes, masks, img_res,confidence,class_name
