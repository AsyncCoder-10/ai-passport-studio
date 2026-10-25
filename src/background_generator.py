import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# Load a lightweight, fast text-to-image pipeline (SD-Turbo)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch_dtype,
    variant="fp16" if device == "cuda" else None,
).to(device)


def generate_studio_background(prompt: str, size=(413, 531)) -> Image.Image:
    """
    Generates a professional studio background.
    Calculates divisible-by-8 dimensions, then resizes back to target size.
    """
    target_w, target_h = size

    # 1. Calculate nearest dimensions divisible by 8
    gen_w = (target_w + 7) // 8 * 8
    gen_h = (target_h + 7) // 8 * 8

    # 2. Generate background at the valid size
    result = pipe(
        prompt,
        num_inference_steps=1,  # SD-Turbo is optimized for 1-step
        guidance_scale=0.0,
        height=gen_h,
        width=gen_w,
    )
    generated_bg = result.images[0]

    # 3. Resize back to the exact passport photo dimensions
    final_bg = generated_bg.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return final_bg
