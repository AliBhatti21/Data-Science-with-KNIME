import sys, copy
import os, json
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
from io import BytesIO
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch

torch.set_num_threads(1)  # Limits CPU threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from utils import knutills as kutil

LOGGER = logging.getLogger(__name__)

knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)

@knext.node(
    name="Yolo Segmentation Node",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/denoise.png",
    category=knimeVis_category,
    id="seg-path"
)
@knext.input_table(name="Input Image", description="Table containing the input image.")
@knext.output_table(name="Segmented Image", description= "Table containing YOLO segmentated Image.")
@knext.output_table(name="Segmented Output Table", description="Table containing YOLO segmentation results including bounding boxes, confidence scores, class IDs, and masks).")

class YOLOSegmentation:
    """    
    YOLO Image Segmentation

    The YOLO Image Segmentation node performs image segmentation using the You Only Look Once (YOLO) model. It accepts a table containing image, and a path of the model, whcih can be downloaded from the ultrlytics https://docs.ultralytics.com/models/yolo11/#key-features ,to processes the images, and outputs a table with the segmentation results along with bounding boxes, confidence scores, class ID, and mask coordinates.
    
    """

    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column containing the image paths or objects.",
        port_index=0,
        column_filter=kutil.is_png,
    )

    # model file path 
    model_path = knext.LocalPathParameter(
        label="Model file path",
        description="Specify the path of the model file to be used here.",
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema):
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]

        if not image_columns:
            raise ValueError("No image columns available for image segmentation.")

        self.image_column = image_columns[-1][0] if image_columns else None

        LOGGER.info(f"Selected image column: {self.image_column}")
        
        if not os.path.isfile(self.model_path):
            LOGGER.error(f"Model file not found at: {self.model_path}")
            raise ValueError("Model file not found at the specified path.")
        
        # Define output schema for Table 1 (Segmented Image)
        output_schema_1 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "img_id"),
            knext.Column(knext.logical(Image.Image), "Segmented Image")
        ])
        
        # Define output schema for Table 2 (Segmentation Results)
        output_schema_2 = knext.Schema.from_columns([
            knext.Column(knext.int32(), "img_id"),      #define image id to uniquly identify the image 
            knext.Column(knext.string(), "Bounding Box"),
            knext.Column(knext.string(), "Confidence Score"),
            knext.Column(knext.string(), "Class ID"),
            knext.Column(knext.string(), "Mask Coordinates"),  # Mask as list of coordinates
        ])
        return output_schema_1, output_schema_2

    def execute(self, exec_context, input_table):
        input_df = input_table.to_pandas()
        try:
            model = YOLO(self.model_path, verbose=True)
            LOGGER.info("YOLO Model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Detect GPU, otherwise use CPU 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        device = torch.device("cpu")  # Force CPU
        model.to(device)  # Move model to CPU

        # Initialize lists to collect results
        segmented_image_data = []  # For Table 1 (Segmented Image)
        segmentation_results = []  # For Table 2 (Segmentation Results)

        # Iterate over all images in the input table
        for img_id, image in enumerate(input_df[self.image_column]):
            # Ensure the image is a tensor with the correct shape
            if isinstance(image, Image.Image):     # Check if it's a PIL image
                transform = transforms.Compose([
                    transforms.Resize((640, 640)),  # Resize to 640x640 as mentioned
                    transforms.ToTensor(),          # Convert to tensor
                ])
                image_tensor = transform(image).unsqueeze(0).to(device)  # image moved to CPU
            else:
                LOGGER.warning(f"Unsupported image format at index {img_id}. Expected PIL image.")
                continue

            # Run YOLO model
            results = model(image_tensor)

            # Lists to store detection results
            bbox_list = []
            conf_list = []
            class_list = []
            mask_list = []

            if results and len(results) > 0:
                result = results[0]
                bboxes = result.boxes.xyxyn.cpu().numpy().tolist()  # Convert to list
                confidences = result.boxes.conf.cpu().numpy().tolist()  # Convert to list
                class_ids = result.boxes.cls.cpu().numpy().tolist()  # Convert to list
                masks = result.masks.xyn if result.masks is not None else []  # List of mask coordinates

                # Convert YOLO results into structured lists
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    confidence = (confidences[i])
                    class_id = (class_ids[i])

                    if masks and len(masks) > 0:
                        mask_coords = [mask.flatten().tolist() for mask in masks]  # Convert masks to JSON string
                    else:
                        mask_coords = None

                    # Append results to lists
                    bbox_list.append((bbox))
                    conf_list.append((confidence))
                    class_list.append((class_id))
                    mask_list.append(mask_coords)

                # Generate segmented image
                segmented_image = Image.fromarray(result.plot()[:, :, ::-1])

                # Append segmented image data to Table 1
                segmented_image_data.append([img_id, segmented_image])

                # Append segmentation results to Table 2
                segmentation_results.append([
                    img_id,
                    json.dumps(bbox_list),
                    json.dumps(conf_list),
                    json.dumps(class_list),
                    json.dumps(mask_list),
                ])

            else:
                LOGGER.warning(f"No segmentation results for image at index {img_id}.")
                # Append empty results for this image
                segmentation_results.append([img_id, json.dumps([]), json.dumps([]), json.dumps([]), json.dumps([])])

        # Create DataFrames for output
        if segmented_image_data:
            segmented_image_df = pd.DataFrame(segmented_image_data, columns=["img_id", "Segmented Image"]).astype({"img_id": "int32"})
        else:
            segmented_image_df = pd.DataFrame(columns=["img_id", "Segmented Image"])

        if segmentation_results:
            segmentation_results_df = pd.DataFrame(segmentation_results, columns=[
                "img_id", "Bounding Box", "Confidence Score", "Class ID", "Mask Coordinates"
            ]).astype({"img_id": "int32"})
        else:
            segmentation_results_df = pd.DataFrame(columns=[
                "img_id", "Bounding Box", "Confidence Score", "Class ID", "Mask Coordinates"
            ])

        # Return both output tables
        return knext.Table.from_pandas(segmented_image_df), knext.Table.from_pandas(segmentation_results_df)