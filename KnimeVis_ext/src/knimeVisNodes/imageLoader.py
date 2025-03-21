import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutills as kutil
from PIL import Image 
import os
import io
import base64

# Logger setup
LOGGER = logging.getLogger(__name__)

# define sub-category here
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)

# Node definition
@knext.node(
    name="Image Loader",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=knimeVis_category,
    id="dl-image",
)
@knext.output_table(
    name="Image Data",
    description="Table containing image in PIL Image format.",
)

@knext.input_table(
    name="Input Table",
    description="Table containing paths to images.",
)

class ImageReader:
    """
    Image Reader Node

    This node loads an image from the specified file path and returns it as a PIL Image.
    """
    # define your parameter
    image_path_column = knext.ColumnParameter(
        label="Image Path Column",
        description="Select the column that contains the paths to the images.",
        port_index=0,
        column_filter=kutil.is_string,
        include_row_key=False,
    )
    
    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        
        # Filter string columns for image paths
        # TODO the reader cannot handle the Path format as input (work on knutils)
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_string(c) ]

        if not image_columns:
            # If no string columns are found, raise an exception or return None
            raise ValueError("No string columns available for image paths.")

        # Log available image columns
        LOGGER.info(f"Available image path columns: {image_columns[-1]}")

        # Set default column if not already set
        if self.image_path_column is None:
            self.image_path_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.warning(f"Selected image path column: {self.image_path_column}")

        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "Image")])

        return output_schema
    
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        df = input_table.to_pandas()
        images_path = df[self.image_path_column]
        LOGGER.warning(f"images_path: {images_path}")

        def read_image(path):
            # This handles different OS path formats
            normalized_path = os.path.normpath(path) 
                    
            # Check if the file exists before attempting to read it
            if not os.path.exists(normalized_path):
                LOGGER.error(f"File does not exist: {normalized_path}")
                return None

            # Read image
            image = Image.open(normalized_path)
            if image is None:
                LOGGER.error(f"Failed to read image at path: {normalized_path}")
                return None

               
            return image
        
        df["Image"] = [read_image(i) for i in images_path]

        return knext.Table.from_pandas(df)