import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutils as kutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image 
import time

# Logger setup
LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = kutil.get_knimeVis_category()

# Node definition
@knext.node(
    name="Denoising",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/denoise.png",
    category=knimeVis_category,
    id="denoise-image",
)
@knext.output_table(
    name="Image Data",
    description="Table containing image",
)

@knext.input_table(
    name="Image Data",
    description="Table containing image",
)

class Denoising:
    """
    Denoising

    This node reduces noise in images by applying advanced filtering techniques, enhancing image clarity and quality for downstream analysis.
    """

    # Define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply Denoising.",
        port_index=0,
        column_filter=kutil.is_png
    )

    class AlgorithmOptions(knext.EnumParameterOptions):
        MEDIAN = ("Median Filtering", "Removes noise by replacing each pixel's value with the median value of the surrounding pixels, preserving edges effectively. Algorithm by OpenCV library")
        GAUSSIAN = ("Gaussian Filtering", "Applies a Gaussian filter to reduce noise. Algorithm by OpenCV library")

    algorithm_selection_param = knext.EnumParameter(
        label="Denoising Algorithm Selection",
        description="Select the algorithm to produce denoised images.",
        default_value=AlgorithmOptions.MEDIAN.name,
        enum=AlgorithmOptions
    )

    filter_size =  knext.IntParameter(
        label="Filter size",
        description="Specify an odd value to define the size of the filter box used for image denoising.",
        default_value=3,
        min_value=3,
        max_value=15,
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

        if self.filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd integer.")
        
        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "Denoised Image")])

        return output_schema
    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        # Start timing
        start_time = time.time()
        
        df = input_table.to_pandas()
        images = df[self.image_column]
        
        # Parallel processing of images
        with ThreadPoolExecutor(max_workers=16) as executor:
            df["Denoised Image"] = list(executor.map(self.process_image, images))

        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        LOGGER.info(f"Node execution completed in {elapsed_time:.2f} seconds.")
            
        return knext.Table.from_pandas(df)
    

    # Helper function to process each image
    def process_image(self, image):
        try:
            if self.algorithm_selection_param == self.AlgorithmOptions.MEDIAN.name:
                # Use median_filter_opencv for MEDIAN option
                return self.median_filter_opencv(image, self.filter_size)
            elif self.algorithm_selection_param == self.AlgorithmOptions.GAUSSIAN_openCV.name:
                return self.gaussian_filter_opencv(image, self.filter_size)
            else:
                raise ValueError(f"Unexpected algorithm: {self.algorithm_selection_param}")
        except Exception as e:
            LOGGER.error(f"Error processing image: {e}")
            return None
            
    def median_filter_opencv(self, image, filter_size):
        # Convert image to NumPy array
        data = np.array(image.convert("L"), dtype=np.uint8)
        # Apply OpenCV median filter
        denoised = cv.medianBlur(data, filter_size)
        # Convert back to PIL image
        return Image.fromarray(denoised)
    
    def gaussian_filter_opencv(self, image, filter_size):
        # Convert image to NumPy array
        data = np.array(image.convert("L"), dtype=np.uint8)
        # Apply Gaussian blur filter
        denoised = cv.GaussianBlur(data, (filter_size, filter_size), 0)
        # Convert back to PIL image
        return Image.fromarray(denoised)
