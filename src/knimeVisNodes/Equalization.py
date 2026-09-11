import json
import logging
import copy
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
from utils import knutils as kutil

LOGGER = logging.getLogger(__name__)

# define sub-category here
knimeVis_category = kutil.get_knimeVis_category()

# Define the KNIME node for Equalization
@knext.node(
    name="Equalization",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/equalized.png",
    category=knimeVis_category,
    id="eq-image"
)
@knext.input_table(name="Input Image Path", description="Table containing the image file.")
@knext.output_table(name="Equalized Image", description="Table containing the equlaized image.")
@knext.output_table(name="Histograms Table", description="Table containing histogram columns of the image.")

class Equalization:
    """    
    Image Histogram Equalization and Analysis

    The Image Histogram Equalization and Analysis node enhances the contrast of grayscale or RGB images through histogram equalization, making it an essential tool for image preprocessing workflows. It accepts a table containing image file paths, processes the images, and outputs a table enriched with detailed insights into the histogram equalization process. This node is particularly useful for computer vision applications, educational demonstrations, and preprocessing steps in machine learning pipelines.

    The output table includes the Equalized Image, showcasing the improved contrast after processing, and three additional columns providing detailed analysis: the Original Histogram of the input image, the Histogram Equalized values post-processing, and the Transfer Function, which maps intensity levels from the original image to the equalized result. These outputs enable users to understand the transformation and its impact on image intensity distribution.

    """
     # define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply spatial filtering.",
        port_index=0,
        column_filter=kutil.is_png
    )


    def configure(
            self,
            configure_context: knext.ConfigurationContext,
            input_schema_1: knext.Schema,
        ):
            
            image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]

            if not image_columns:
                raise ValueError("No string columns available for image paths.")
            
            LOGGER.info(f"Available image path columns: {image_columns[-1]}")

            if self.image_column is None:
                self.image_column = image_columns[-1][0]
        
            LOGGER.warning(f"Selected image path column: {self.image_column}")
            
            # Define output schemas for both output tables
            # output_schema_1 = knext.Schema.from_columns([
            #     knext.Column(knext.logical(Image.Image), "Equalized Image")
            # ])

            # TODO setted for demo
            output_schema_1 = input_schema_1.append(
                [knext.Column(knext.logical(Image.Image), "Equalized Image")])
            
            output_schema_2 = knext.Schema.from_columns([
                
                knext.Column(knext.string(), "Original Histogram"),
                knext.Column(knext.string(), "Histogram Equalized"),
                knext.Column(knext.string(), "Transfer Function"),
            ])
            
            
            
            # Return both output schemas
            return output_schema_1, output_schema_2

    def execute(self, exec_context, input_table):
        if hasattr(input_table, "to_pandas"):
            input_df = input_table.to_pandas()
        elif isinstance(input_table, pd.DataFrame):
            input_df = input_table
        elif isinstance(input_table, dict):
            input_df = pd.DataFrame.from_dict(input_table)
        else:
            raise TypeError(f"Unexpected input_table type: {type(input_table)}")
        
        input_df2 = copy.deepcopy(input_df)
        Original_hist = []
        hist_equalize = []
        transfer_functions = []

        def imhist(im):
            m, n = im.shape
            h = [0.0] * 256
            for i in range(m):
                for j in range(n):
                    h[im[i, j]] += 1
            return np.array(h) / (m * n)

        def cumsum(h):
            return [sum(h[:i + 1]) for i in range(len(h))]

        def histeq(im):
            h = imhist(im)
            cdf = np.array(cumsum(h))
            sk = np.uint8(255 * cdf)
            s1, s2 = im.shape
            Y = np.zeros_like(im)
            for i in range(s1):
                for j in range(s2):
                    Y[i, j] = sk[im[i, j]]
            H = imhist(Y)
            return Y, h, H, sk

        def process_images(image_paths):
            for image_path in image_paths:
                image = np.array(image_path.convert("L"))
                if image.ndim == 3:
                    image = np.uint8(0.2126 * image[:, :, 0] +
                                    0.7152 * image[:, :, 1] +
                                    0.0722 * image[:, :, 2])

                equalized_image, h, new_h, sk = histeq(image)

                Original_hist.append(h.tolist())
                hist_equalize.append(new_h.tolist())
                transfer_functions.append(sk.tolist())

                yield equalized_image, h.tolist(), new_h.tolist(), sk.tolist()

        equalized_images = []
        original_histograms = []
        hist_equalized = []
        transfer_functions_list = []

        for image, hist, eq_hist, tf in process_images(input_df[self.image_column]):
            equalized_images.append(Image.fromarray(image))
            original_histograms.append(json.dumps(hist))
            hist_equalized.append(json.dumps(eq_hist))
            transfer_functions_list.append(json.dumps(tf))

        input_df["Equalized Image"] = equalized_images
        input_df2["Original Histogram"] = original_histograms
        input_df2["Histogram Equalized"] = hist_equalized
        input_df2["Transfer Function"] = transfer_functions_list
        
        # Create output tables for both output ports
        output_df_2 = input_df2[[ "Original Histogram", "Histogram Equalized", "Transfer Function"]]
        output_df_1 = input_df["Equalized Image"]  # Adjust column names as needed

        # Return both output tables
        return knext.Table.from_pandas(input_df),knext.Table.from_pandas(output_df_2)

    
    