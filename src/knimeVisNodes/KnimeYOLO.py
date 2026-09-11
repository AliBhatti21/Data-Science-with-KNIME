import knime.extension as knext
import numpy as np
import logging
from utils import knutils as kutil
from ultralytics import YOLO
from PIL import Image 
import pandas as pd
import torch
from torchvision import transforms
import os, sys

torch.set_num_threads(1)  # Limits CPU threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Logger setup
LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = kutil.get_knimeVis_category()

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
    description="Contains bounding box coordinates for each detected object.",
)
@knext.output_table(
    name="Segmentation Masks",
    description="Contains the segmentation masks for each detected object.",
)
@knext.output_table(
    name="Image Masks",
    description="Contains the input image overlaid with segmentation masks for visualization.",
)

@knext.input_table(
    name="Image Data",
    description="Table containing the image(s) to be segmented.",
)

class KnimeYOLO:
    """
    KnimeYOLO

    Perform segmentation, object detection, and classification using a YOLO model.

    This KnimeNode applies a pretrained YOLO model on input images based on the provided model weights.
    It supports both official pretrained models available from Ultralytics
    (https://docs.ultralytics.com/tasks/segment/) and custom fine-tuned models.
    """

    # Define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column containing PNG images on which YOLO segmentation and detection will be applied.",
        port_index=0,
        column_filter=kutil.is_png
    )

    image_id = knext.ColumnParameter(
        label="Image ID",
        description="Select the column to use as a unique identifier for each image.",
        port_index=0
    )

    
    # Define the file path parameter
    model_path = knext.LocalPathParameter(
        label="Model file path",
        description="Specify the local file path to the YOLO model weights (.pt file).",
    )

    class DeviceOptions(knext.EnumParameterOptions):
        CPU = ("cpu", "Use CPU for inference")
        GPU = ("gpu", "Use GPU for inference if CUDA is available")

    device: str = knext.EnumParameter(
        label="Computation Device",
        description="Select the device for running model inference: CPU or GPU (CUDA).",
        default_value=DeviceOptions.CPU.name,
        enum=DeviceOptions
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
            knext.Column(knext.double(), "x_center"),  
            knext.Column(knext.double(), "y_center"),  
            knext.Column(knext.double(), "width"),  
            knext.Column(knext.double(), "height"),
            knext.Column(knext.string(), "class"),  
            knext.Column(knext.double(), "confidences")  
        ])
        output_schema_masks = knext.Schema.from_columns([
            knext.Column(knext.string(), "img_id"),          
            knext.Column(knext.logical(Image.Image), "masks"),
            knext.Column(knext.string(), "class"),  
            knext.Column(knext.double(), "confidences") 
        ])  

        output_schema = input_schema_1.append(
            [ knext.Column(knext.logical(Image.Image), "ImageMasked")])

        # Return the output schemas
        return output_schema_boxes, output_schema_masks, output_schema

    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Determine and log device
        device = "cuda" if self.device == "CUDA" and torch.cuda.is_available() else "cpu"
        LOGGER.info(f"Using device: {device}")

        # Load a pretrained YOLO model
        model = YOLO(self.model_path)
        model.to(device)

        df = input_table.to_pandas()
        
        # Initialize lists to collect results
        boxes_data = {"img_id": [], "x_center": [], "y_center": [], "width": [], "height": [], "class": [], "confidence": []}
        masks_data = {"img_id": [], "masks": [], "class": [], "confidence": []}

        for idx in df.index:
            img = df.at[idx,self.image_column]  # Get image path or data
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_id = df.at[idx,self.image_id] # Use index as img_id (or replace with a column like row["id"])

            # Process image with YOLO model
            boxes, mask_images, img_res, confidences, class_names = self.process_image(model, img, device=device)

            if boxes is not None and len(boxes) > 0:
                # Convert tensor to numpy if needed
                 for i, single_box in enumerate(boxes):  # Iterate over each box (e.g., [x_min, y_min, x_max, y_max])
                    boxes_data["img_id"].append(img_id)
                    boxes_data["x_center"].append(single_box[0])  # Convert to list for schema
                    boxes_data["y_center"].append(single_box[1])
                    boxes_data["width"].append(single_box[2])
                    boxes_data["height"].append(single_box[3])
                    boxes_data["class"].append(class_names[i] if i < len(class_names) else "unknown")
                    boxes_data["confidence"].append(confidences[i] if i < len(confidences) else -1)

            
            if mask_images is not None and len(mask_images) > 0:
                # Convert tensor to numpy if needed
                for i, single_mask in enumerate(mask_images):  # Iterate over each mask
                    masks_data["img_id"].append(img_id)
                    masks_data["masks"].append(single_mask)  # Convert to list for schema
                    masks_data["class"].append(class_names[i] if i < len(class_names) else "unknown")
                    masks_data["confidence"].append(confidences[i] if i < len(confidences) else -1)

            # Append the processed image with mask to the DataFrame
            df.at[idx, "ImageMasked"] = img_res
            

        # Create output DataFrames
        boxes_df = pd.DataFrame(boxes_data)
        masks_df = pd.DataFrame(masks_data)
        # df["ImageMasked"] = img_res

        output_schema_boxes = knext.Table.from_pandas(boxes_df)
        output_schema_masks = knext.Table.from_pandas(masks_df)
        output_schema = knext.Table.from_pandas(df)

        # Convert to KNIME tables
        return output_schema_boxes, output_schema_masks, output_schema


    
    
    def process_image(self, model, img, device="cpu"):
        result = model.predict(img, device=0 if device == "cuda" else "cpu")  # Example: result could be a list or dict

        # Get bounding boxes coordinates
        try:
            boxes = result[0].boxes.xywhn #Contains normalized [x_center, y_center, width, height] coordinates relative to the original image dimensions (values between 0-1)
            boxes = boxes.numpy()
        except:
            LOGGER.info(f"Bounding boxes not available")
            boxes = np.zeros((1, 4))  

        # Get segmentation mask 
        try:
            mask_images = []
            for i, mask_tensor in enumerate(result[0].masks.data):
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)  # Convert to uint8
                mask_img = Image.fromarray(mask_np, mode='L')  # B/W mask image
                mask_images.append(mask_img)

        except Exception as e:
            LOGGER.info(f"Masks not available: {str(e)}")
            mask_images = None  # Empty array for masks (no coordinates)

        # Get class IDs and confidences
        try:
            class_ids = result[0].boxes.cls.cpu().numpy().astype(int)  # Integer class indices
            confidences = result[0].boxes.conf.cpu().numpy()  # Confidence scores
            class_names = [model.names[c] for c in class_ids]  # Class names from model
        except Exception as e:
            LOGGER.info(f"Classes/confidences not available: {str(e)}")
            class_ids, confidences, class_names = [], [], []
        
        img_res = Image.fromarray(result[0].plot()[:, :, ::-1])

        return boxes, mask_images, img_res, confidences, class_names
