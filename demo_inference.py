from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import argparse

def run_hf_inference(image_path, model_name="starvector/starvector-8b-im2svg"):
    print(f"\nRunning HuggingFace inference with {model_name}...")
    
    # Load model
    starvector = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    processor = starvector.model.processor
    tokenizer = starvector.model.svg_transformer.tokenizer

    starvector.cuda()
    starvector.eval()

    # Load and process image
    image_pil = Image.open(image_path)
    image = processor(image_pil, return_tensors="pt")['pixel_values'].cuda()
    if not image.shape[0] == 1:
        image = image.squeeze(0)
    batch = {"image": image}

    # Generate SVG
    raw_svg = starvector.generate_im2svg(batch, max_length=4000)[0]
    svg, raster_image = process_and_rasterize_svg(raw_svg)
    
    return svg, raster_image

def run_native_inference(image_path, model_name="starvector/starvector-1b-im2svg"):
    print(f"\nRunning native StarVector inference with {model_name}...")
    
    # Load model
    starvector = StarVectorForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16
    )
    starvector.cuda()
    starvector.eval()

    # Load and process image
    image_pil = Image.open(image_path)
    image_pil = image_pil.convert('RGB')
    image = starvector.process_images([image_pil])[0].to(torch.float16).cuda()
    batch = {"image": image}

    # Generate SVG with some parameters for better results
    raw_svg = starvector.generate_im2svg(
        batch, 
        max_length=4000,
        temperature=1.5,
        length_penalty=-1,
        repetition_penalty=3.1
    )[0]
    svg, raster_image = process_and_rasterize_svg(raw_svg)
    
    return svg, raster_image

def save_results(svg, raster_image, prefix):
    # Save SVG
    with open(f"{prefix}_output.svg", "w") as f:
        f.write(svg)
    
    # Save rasterized image
    raster_image.save(f"{prefix}_rasterized.png")
    
    print(f"Results saved as {prefix}_output.svg and {prefix}_rasterized.png")

def main():
    parser = argparse.ArgumentParser(description='Run StarVector inference demo')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model-8b', type=str, default="starvector/starvector-8b-im2svg", help='8B model name/path')
    parser.add_argument('--model-1b', type=str, default="starvector/starvector-1b-im2svg", help='1B model name/path')
    args = parser.parse_args()

    # Run both inference methods
    svg_hf, raster_hf = run_hf_inference(args.image, args.model_8b)
    save_results(svg_hf, raster_hf, "hf")
    
    # Clear CUDA cache between runs
    torch.cuda.empty_cache()
    
    # svg_native, raster_native = run_native_inference(args.image, args.model_1b)
    # save_results(svg_native, raster_native, "native")

if __name__ == "__main__":
    main() 