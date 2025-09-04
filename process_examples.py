from PIL import Image
import torch
import gc
import os
from transformers import AutoModelForCausalLM
from starvector.data.util import process_and_rasterize_svg
import base64
from io import BytesIO
import sys

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    # Initialize model
    print("Initializing model...")
    model_name = "starvector/starvector-8b-im2svg"
    
    try:
        # Clear CUDA cache before loading model
        torch.cuda.empty_cache()
        gc.collect()
        
        starvector = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map="auto"  # This will handle memory better
        )
        processor = starvector.model.processor
        
        # Move to GPU and set eval mode
        starvector.eval()
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Start HTML file
    html_content = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 10px; border: 1px solid black; text-align: center; }
            img { max-width: 300px; height: auto; }
            pre { white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <h1>StarVector PNG to SVG Results</h1>
        <table>
            <tr>
                <th>Original PNG</th>
                <th>Generated SVG (Rasterized)</th>
                <th>Raw SVG Code</th>
            </tr>
    """
    
    # Process each PNG in the examples directory
    example_dir = 'assets/examples'
    png_files = sorted([f for f in os.listdir(example_dir) if f.endswith('.png')])
    
    print(f"Found {len(png_files)} PNG files to process")
    
    for i, png_file in enumerate(png_files, 1):
        print(f"\nProcessing {png_file} ({i}/{len(png_files)})...")
        
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load and process image
            image_path = os.path.join(example_dir, png_file)
            image_pil = Image.open(image_path)
            
            # Generate SVG
            print("  Generating SVG...")
            image = processor(image_pil, return_tensors="pt")['pixel_values'].to(starvector.device)
            if not image.shape[0] == 1:
                image = image.squeeze(0)
            batch = {"image": image}
            
            raw_svg = starvector.generate_im2svg(batch, max_length=4000)[0]
            print("  Rasterizing SVG...")
            svg, raster_image = process_and_rasterize_svg(raw_svg)
            
            # Save SVG to file
            svg_filename = os.path.join('output', png_file.replace('.png', '.svg'))
            with open(svg_filename, 'w') as f:
                f.write(svg)
            print(f"  Saved SVG to {svg_filename}")
            
            # Convert images to base64
            orig_b64 = image_to_base64(image_pil)
            raster_b64 = image_to_base64(raster_image)
            
            # Add to HTML
            html_content += f"""
            <tr>
                <td>
                    <img src="data:image/png;base64,{orig_b64}">
                    <br>
                    <small>{png_file}</small>
                </td>
                <td>
                    <img src="data:image/png;base64,{raster_b64}">
                </td>
                <td>
                    <pre style="text-align: left; max-height: 200px; overflow: auto;">
                        {svg.replace('<', '&lt;').replace('>', '&gt;')}
                    </pre>
                </td>
            </tr>
            """
            
        except Exception as e:
            print(f"Error processing {png_file}: {str(e)}")
            html_content += f"""
            <tr>
                <td colspan="3">Error processing {png_file}: {str(e)}</td>
            </tr>
            """
            
        # Save HTML after each image in case of crashes
        with open('output/results.html', 'w') as f:
            f.write(html_content + "</table></body></html>")
    
    # Close HTML
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Save final HTML report
    with open('output/results.html', 'w') as f:
        f.write(html_content)
    
    print("\nProcessing complete! Results saved to output/results.html")

if __name__ == "__main__":
    main() 