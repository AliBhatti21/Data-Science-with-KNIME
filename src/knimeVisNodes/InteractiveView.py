import sys
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import knutils as kutil
import json
import base64
from io import BytesIO

torch.set_num_threads(1)  # Limits CPU threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LOGGER = logging.getLogger(__name__)

# Define sub-category here
knimeVis_category = kutil.get_knimeVis_category ()

@knext.node(
    name="Interactive Prompt (bbox)",
    node_type=knext.NodeType.VISUALIZER,
    icon_path="icons/sam_icon.png",
    category=knimeVis_category,
    id="draw-bbox"
)
@knext.input_table(
    name="Input Images", 
    description="Select the column containing the input images"
)
@knext.output_view(
    name="Interactive widget", 
    description="Draw the region of interests (RoIs) in the interactive mode"
)
class SAMBBoxDrawerNode:
    """
    Interactive bounding box drawing tool with per-image indexing.
    Allows downloading box coordinates as CSV with proper image-based indexing.
    """
    
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column containing the image objects.",
        port_index=0,
        column_filter=kutil.is_png,
    )

    def __init__(self):
        self.images_base64 = []
        self.image_dimensions = []

    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema):
        image_columns = [c.name for c in input_schema_1 if kutil.is_png(c)]
        if not image_columns:
            raise ValueError("No image columns available in input table")
        self.image_column = image_columns[0]
        return None

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        if len(df) == 0:
            raise ValueError("Input table is empty")
            
        self.images_base64 = []
        self.image_dimensions = []
        
        for img in df[self.image_column]:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            self.images_base64.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            self.image_dimensions.append((img.width, img.height))
        
        return knext.view(self._generate_interactive_html())

    def _generate_interactive_html(self):
        img_data_js = json.dumps([
            {"src": f"data:image/png;base64,{img}", "width": w, "height": h}
            for img, (w, h) in zip(self.images_base64, self.image_dimensions)
        ])
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .container {{
            max-width: 100%;
            overflow: auto;
        }}
        #canvas-container {{
            position: relative;
            display: inline-block;
            border: 2px solid #333;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            margin-bottom: 15px;
        }}
        #canvas {{
            display: block;
            cursor: crosshair;
            background-color: #f0f0f0;
            max-width: 100%;
        }}
        .controls {{
            margin: 15px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        button {{
            padding: 8px 15px;
            background-color: #4285f4;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            min-width: 120px;
        }}
        #image-nav {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }}
        #image-counter {{
            font-weight: bold;
            min-width: 100px;
            text-align: center;
        }}
        #resultContainer {{
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
        }}
        a.download-link {{
            display: inline-block;
            padding: 8px 15px;
            background-color: #34a853;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .loading {{
            color: #666;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2 style="color: #4285f4;">Draw Regions of Interest</h2>
        <p>Click and drag to draw bounding boxes (minimum size: 5×5 pixels)</p>
        
        <div id="image-nav">
            <button id="prevBtn" disabled>Previous</button>
            <div id="image-counter">Image 1 of {len(self.images_base64)}</div>
            <button id="nextBtn" {'' if len(self.images_base64) > 1 else 'disabled'}>Next</button>
        </div>
        
        <div id="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
        
        <div class="controls">
            <button id="clearBtn">Clear Current Image</button>
            <button id="generateBtn">Generate CSV</button>
        </div>
        
        <div id="resultContainer">
            <p class="loading">Loading first image...</p>
        </div>
    </div>
    
    <script>
        // Image data
        const images = {img_data_js};
        let currentImageIndex = 0;
        let boxesByImage = Array(images.length).fill().map(() => []);
        
        // DOM elements
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const resultContainer = document.getElementById('resultContainer');
        const imageCounter = document.getElementById('image-counter');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        // Drawing state
        let isDrawing = false;
        let startPos = {{ x: 0, y: 0 }};
        let currentImage = new Image();
        
        // Initialize
        loadImage(currentImageIndex);
        
        function loadImage(index) {{
            if (index < 0 || index >= images.length) return;
            
            // Update navigation buttons
            prevBtn.disabled = index <= 0;
            nextBtn.disabled = index >= images.length - 1;
            
            currentImageIndex = index;
            const imageData = images[index];
            
            // Show loading state
            resultContainer.innerHTML = '<p class="loading">Loading image...</p>';
            
            currentImage.onload = function() {{
                // Set canvas dimensions
                canvas.width = imageData.width;
                canvas.height = imageData.height;
                
                // Draw the image
                ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
                
                // Update UI
                imageCounter.textContent = `Image ${{index + 1}} of ${{images.length}}`;
                resultContainer.innerHTML = '<p style="color:green;">Ready to draw - click and drag on the image</p>';
                
                // Redraw existing boxes
                redraw();
            }};
            
            currentImage.onerror = function() {{
                resultContainer.innerHTML = '<p style="color:red;">Error loading image</p>';
            }};
            
            currentImage.src = imageData.src;
        }}
        
        function getCurrentBoxes() {{
            return boxesByImage[currentImageIndex];
        }}
        
        function setCurrentBoxes(newBoxes) {{
            boxesByImage[currentImageIndex] = newBoxes;
        }}
        
        function getMousePos(canvas, evt) {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {{
                x: Math.round((evt.clientX - rect.left) * scaleX),
                y: Math.round((evt.clientY - rect.top) * scaleY)
            }};
        }}
        
        function drawBox(box, style = 'blue', lineWidth = 3, dashed = false) {{
            ctx.beginPath();
            ctx.rect(box.x1, box.y1, box.x2-box.x1, box.y2-box.y1);
            ctx.strokeStyle = style;
            ctx.lineWidth = lineWidth;
            if (dashed) {{
                ctx.setLineDash([5, 5]);
            }} else {{
                ctx.setLineDash([]);
            }}
            ctx.stroke();
        }}
        
        function redraw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (currentImage.complete) {{
                ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            }}
            getCurrentBoxes().forEach(box => drawBox(box));
        }}
        
        // Navigation handlers
        prevBtn.addEventListener('click', () => loadImage(currentImageIndex - 1));
        nextBtn.addEventListener('click', () => loadImage(currentImageIndex + 1));
        
        // Canvas handlers
        canvas.addEventListener('mousedown', (e) => {{
            const pos = getMousePos(canvas, e);
            startPos = pos;
            isDrawing = true;
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (!isDrawing) return;
            const pos = getMousePos(canvas, e);
            redraw();
            drawBox({{
                x1: startPos.x,
                y1: startPos.y,
                x2: pos.x,
                y2: pos.y
            }}, 'blue', 2, true);
        }});
        
        canvas.addEventListener('mouseup', (e) => {{
            if (!isDrawing) return;
            const pos = getMousePos(canvas, e);
            const width = Math.abs(pos.x - startPos.x);
            const height = Math.abs(pos.y - startPos.y);
            
            if (width >= 5 && height >= 5) {{
                const currentBoxes = getCurrentBoxes();
                currentBoxes.push({{
                    x1: Math.min(startPos.x, pos.x),
                    y1: Math.min(startPos.y, pos.y),
                    x2: Math.max(startPos.x, pos.x),
                    y2: Math.max(startPos.y, pos.y)
                }});
                setCurrentBoxes(currentBoxes);
                redraw();
            }}
            isDrawing = false;
        }});
        
        
        document.getElementById('generateBtn').addEventListener('click', () => {{
            const allBoxes = [];
            let globalIndex = 0;
            
            boxesByImage.forEach((boxes, imgIndex) => {{
                boxes.forEach((box, boxIndex) => {{
                    allBoxes.push({{
                        imgIndex,
                        boxIndex,
                        globalIndex: globalIndex++,
                        x1: box.x1,
                        y1: box.y1,
                        x2: box.x2,
                        y2: box.y2
                    }});
                }});
            }});
            
            if (allBoxes.length === 0) {{
                resultContainer.innerHTML = '<p style="color:red;">Please draw at least one box on any image</p>';
                return;
            }}
            
            const csvContent = [
                "Image Index,BBox Index,Global Box Index,X_min,Y_min,X_max,Y_max",
                ...allBoxes.map(box => 
                    `${{box.imgIndex}},${{box.boxIndex}},${{box.globalIndex}},${{box.x1}},${{box.y1}},${{box.x2}},${{box.y2}}`
                )
            ].join("\\n");
            
            const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "bounding_boxes.csv");
            link.className = "download-link";
            link.textContent = "Download All Bounding Boxes CSV";
            
            resultContainer.innerHTML = `
                <div style="margin-bottom: 15px;">
                    <p style="color:green; font-weight: bold;">
                        Generated CSV for ${{allBoxes.length}} box(es) across ${{images.length}} image(s)
                    </p>
                    <p>Box coordinates preview (first 5 boxes):</p>
                    <pre style="background: #fff; padding: 10px; border-radius: 4px; overflow: auto; max-height: 200px;">
${{JSON.stringify(allBoxes.slice(0, 5), null, 2)}}
                    </pre>
                    ${{allBoxes.length > 5 ? `<p>... and ${{allBoxes.length - 5}} more boxes</p>` : ''}}
                </div>
            `;
            resultContainer.appendChild(link);
        }});
    </script>
</body>
</html>
"""