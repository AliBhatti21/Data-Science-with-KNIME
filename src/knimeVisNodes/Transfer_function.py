import json
import logging
import numpy as np
import pandas as pd
import knime.extension as knext
import matplotlib.pyplot as plt
import seaborn as sns
from utils import knutils as kutil

# Logger setup
LOGGER = logging.getLogger(__name__)

# define sub-category here
knimeVis_category = kutil.get_knimeVis_category()

@knext.node(
    name="Equalization Visualizer",
    node_type=knext.NodeType.VISUALIZER,
    icon_path="icons/visualizations.png",
    category=knimeVis_category,
    id = "vl-image"
)
@knext.input_table(
    name="Input Table", 
    description="Table containing histogram equalization data."
    )
@knext.output_view(
    name="Output View", 
    description="Seaborn visualization of the selected column."
    )
class VisualizerNode:
    
    """
    Histogram Transformation and Visualization

    This node provides a comprehensive visualization of histogram equalization. This node provides you a view of the original image, the equalized image, and the corresponding transfer function (mapping from original to equalized intensity levels). This visualization facilitates better understanding of the matrix and evaluation of the histogram equalization and its impact on image contrast enhancement. 

    """
    input_column = knext.ColumnParameter(
        "Input Column",
        "Select the column containing the data for visualization.",
        include_none_column=False,
        column_filter=kutil.is_string
    )

    def configure(self, config_context, input_table_schema):
        
        string_columns = [(c.name, c.ktype) for c in input_table_schema if kutil.is_string(c)]

        if not string_columns:
            raise ValueError("String column does not exist in the input table.")
        
        LOGGER.info(f"Available string column: {string_columns[-1]}")

        # Ensure the input table has at least one column
        if self.input_column is None:
            self.input_column = string_columns[-1][0]

        # Validate that the selected column exists in the input table
        if self.input_column not in input_table_schema.column_names:
            raise ValueError(f"Selected column '{self.input_column}' does not exist in the input table.")
        return None

    def execute(self, exec_context, input_table):
        # Convert KNIME input table to pandas DataFrame
        if hasattr(input_table, "to_pandas"):
            input_df = input_table.to_pandas()
        else:
            raise TypeError("Input data must be convertible to a pandas DataFrame.")

        # Ensure the input table is not empty
        if input_df.empty:
            raise ValueError("Input table is empty. Cannot generate visualization.")

        # Retrieve data from the selected column - first json matrix
        selected_column_data = input_df[self.input_column].iloc[0]
        parsed_data = np.array(json.loads(selected_column_data), dtype=np.uint8)

        # Create the plot dynamically based on the selected column
        plt.figure(figsize=(10, 6))

        if "Transfer Function" in self.input_column:
            # Create a transfer function plot
            data = pd.DataFrame({
                "Input Intensity": np.arange(len(parsed_data)),
                "Output Intensity": parsed_data,
            })
            sns.lineplot(x="Input Intensity", y="Output Intensity", data=data)
            plt.title("Transfer Function")
            plt.ylabel("Input Intensity")
            plt.xlabel("Output Intensity")
            plt.grid(True)

        elif "Histogram Equalized" in self.input_column:
            # Create a histogram plot for equalized data
            data_min = parsed_data.min()
            data_max = parsed_data.max()
            plt.hist(parsed_data.flatten(), bins=50, range=(data_min, data_max), color="blue", alpha=0.7)

            #plt.hist(parsed_data.flatten(), bins=50, range=(0, 1), color="blue", alpha=0.7)
            plt.title("Histogram Equalized")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.grid(True)

        elif "Original Histogram" in self.input_column:
            # Create a histogram plot for the original data
            data_min = parsed_data.min()
            data_max = parsed_data.max()
            plt.hist(parsed_data.flatten(), bins=50, range=(data_min, data_max), color="blue", alpha=0.7)
            #plt.hist(parsed_data.flatten(), bins=50, range=(0, 1), color="blue", alpha=0.7)
            plt.title("Original Histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.grid(True)

        else:
            raise ValueError(f"Unsupported column '{self.input_column}' for visualization.")

        
        return knext.view_seaborn()
    