import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutils as kutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image 

# Logger setup
LOGGER = logging.getLogger(__name__)

# define sub-category here
knimeVis_category = kutil.get_knimeVis_category()

# Node definition
@knext.node(
    name="Edge Detection",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/filterIcon.png",
    category=knimeVis_category,
    id="edgdet-image",
)
@knext.output_table(
    name="Image Data",
    description="Table containing image",
)

@knext.input_table(
    name="Image Data",
    description="Table containing image",
)

class EdgeDetection:
    """
    Edge detection Node

    The Spatial Filtering Node is a custom KNIME node designed to perfrom Edge detection. 
    
    Edge detection involves identifying and locating sharp discontinuities in an image, which correspond to significant changes in intensity or color. 
    These discontinuities are referred to as edges, essential for understanding the structure and content of an image.
    """

    # define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply spatial filtering.",
        port_index=0,
        column_filter=kutil.is_png
    )

    threshold =  knext.IntParameter(
        label="Threshold",
        description="Threshold separates objects from the background in an image by setting a pixel intensity cutoff. Pixels above the threshold are typically classified as foreground (e.g., object or feature of interest), while those below are classified as background",
        default_value=30,
        min_value=3,
        max_value=250,
    )

    class AlgorithmOptions(knext.EnumParameterOptions):
        SOBEL = ("Sobel", "The Sobel algorithm detects edges in images by calculating gradient magnitudes using convolution with 3x3 kernels, highlighting areas of high intensity change for effective edge detection.")
        SOBEL_OpenCV = ("SOBEL_OpenCV","The Sobel algorithm detects edges in images by calculating gradient magnitudes using convolution with 3x3 kernels, highlighting areas of high intensity change for effective edge detection. The algoritm is impemented in the OpenCV library")
        LAPLACE = ("Laplace", "The Laplacian operator detects edges in images by calculating second-order derivatives, highlighting rapid intensity changes. It measures how much the average value of a function around a point deviates from the value at that point, indicating regions of local maxima or minima.")
        ROBERT = ("Robert", "The Roberts operator detects edges by calculating gradients using diagonal 2x2 convolution kernels, highlighting sharp intensity changes, and is efficient for real-time applications despite sensitivity to noise.")

    # TODO add parameter depending on the selected algorithm
    algorithm_selection_param = knext.EnumParameter(
        label="Spatial Filtering Algorithm Selection",
        description="Select the algorithm to produce spatial filtering",
        default_value=AlgorithmOptions.SOBEL.name,
        enum=AlgorithmOptions)
    
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

        # Log available image columns
        LOGGER.info(f"Available image columns: {image_columns[-1]}")

        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.warning(f"Selected image column: {self.image_column}")

        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "EdgeDetectedImage")])

        return output_schema
    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        df = input_table.to_pandas()
        images = df[self.image_column]

        # Parallel processing of images
        with ThreadPoolExecutor(max_workers=16) as executor:
            df["EdgeDetectedImage"] = list(executor.map(self.process_image, images))
  
        return knext.Table.from_pandas(df)
    


    def process_image(self,image):
        try:
            if self.algorithm_selection_param == self.AlgorithmOptions.ROBERT.name:
                return self.robert(image)
            elif self.algorithm_selection_param == self.AlgorithmOptions.SOBEL.name:
                return self.sobel(image)
            elif self.algorithm_selection_param == self.AlgorithmOptions.SOBEL_OpenCV.name:
                return self.sobel_opencv(image)
            elif self.algorithm_selection_param == self.AlgorithmOptions.LAPLACE.name:
                return self.laplace(image)
            else:
                raise ValueError(f"Unexpected algorithm: {self.algorithm_selection_param}")
        except Exception as e:
            LOGGER.error(f"Error processing image: {e}")
            return None
            

    # HACK to work with PIL
    def robert(self,img):
        img = np.array(img.convert("L"), dtype=np.float32)
        r, c = img.shape
        edge_r= np.zeros((r,c))
        edge_l= np.zeros((r,c))
        r_sunnzi = [[1,0],[0,-1]]
        l_sunnzi = [[0,1],[-1,0]]
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild = img[x:x+2, y:y+2]
                    list_robert = r_sunnzi*imgChild
                    edge_r[x, y] = abs(list_robert.sum()) # sum and absolute value

        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild = img[x:x+2, y:y+2]
                    list_robert = l_sunnzi*imgChild
                    edge_l[x, y] = abs(list_robert.sum()) # sum and absolute value
        new_image = np.sqrt(edge_r**2+edge_l**2)
        new_image = new_image/np.max(new_image)*255
        new_image = np.where(new_image >= self.threshold, 255, 0).astype(np.uint8)
        #new_image = np.clip(new_image, np.min(new_image), self.threshold).astype(np.uint8)
        return Image.fromarray(new_image)
                    
    # # The implementation of sobel operator
    def sobel(self,img):
        img = np.array(img.convert("L"), dtype=np.float32)
        r, c = img.shape
        new_image = np.zeros((r, c))
        new_imageX = np.zeros(img.shape)
        new_imageY = np.zeros(img.shape)
        s_suanziX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # X direction
        s_suanziY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])     
        for i in range(r-2):
            for j in range(c-2):
                new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
                new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
                new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5
        new_image=(new_image / np.max(new_image) * 255).astype(np.uint8)
        new_image = np.where(new_image >= self.threshold, 255, 0).astype(np.uint8)
        #new_image = np.clip(new_image, np.min(new_image), self.threshold).astype(np.uint8)
        return Image.fromarray(new_image)
    
    # Laplace operator
    def laplace(self,img):
        img = np.array(img.convert("L"), dtype=np.float32)
        r, c = img.shape
        new_image = np.zeros((r, c))
        L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])          
        for i in range(r-2):
            for j in range(c-2):
                new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))

        new_image = np.abs(new_image) 
        new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image)) * 255
        #new_image = new_image/np.max(new_image)*255      
        #new_image = np.clip(new_image, np.min(new_image), self.threshold).astype(np.uint8)
        new_image = np.where(new_image >= self.threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(new_image)



    def sobel_opencv(self, image):
        # Convert image to grayscale
        gray_image = np.array(image.convert("L"), dtype=np.uint8)

        # Apply Sobel filter in both X and Y directions
        sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude (sqrt(sobel_x^2 + sobel_y^2))
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 8-bit image
        gradient_magnitude = np.clip(gradient_magnitude / np.max(gradient_magnitude) * 255, 0, 255).astype(np.uint8)

        # Apply thresholding to keep only significant edges (adjust threshold as needed)
        thresholded_image = np.clip(gradient_magnitude, np.min(gradient_magnitude), self.threshold).astype(np.uint8)
        ## TODO check thresholding for opencv
        #_, thresholded_image = cv.threshold(gradient_magnitude, 50, 255, cv.THRESH_BINARY)

        # Convert back to an image
        return Image.fromarray(thresholded_image)
    