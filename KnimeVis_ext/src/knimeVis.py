import knime.extension as knext


main_category = knext.category(
    path="/community/",
    level_id="unibz",
    name="UniBZ",
    description="Category for Nodes by the unibz KNIME dev team",
    icon="icons/unibz_icon64.png",
)


#from knimeVisNodes import imageProcessing, imageProcessing2
from knimeVisNodes import imageLoader,Denoising,EdgeDetection,Equalization,IoU,KnimeYOLO,SAM,IoU 
