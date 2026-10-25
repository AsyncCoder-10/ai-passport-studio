import replicate
import requests
from PIL import Image
import io
import os
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch
import numpy as np

# Load the clothing segmentation model (SegFormer)
processor = AutoImageProcessor.from_pretrained(
    "mattmdjaga/segformer_b2_clothes", backend="pil"
)
model = AutoModelForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)


def create_clothing_mask(image: Image.Image) -> Image.Image:
    """Generates a binary mask of the clothing region."""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Resize logits to original image size
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = logits.argmax(dim=1)[0].cpu().numpy()

    # Class indices: 4 = Upper-clothes, 5 = Skirt, 6 = Pants, 7 = Dress
    clothing_classes = [4, 5, 6, 7]
    mask_array = np.isin(pred_seg, clothing_classes).astype(np.uint8) * 255

    mask_image = Image.fromarray(mask_array, mode="L")
    return mask_image


def suit_up_clothing(
    foreground_image: Image.Image, replicate_api_token: str
) -> Image.Image:
    """Uses Replicate API to inpaint the clothing into a professional suit."""
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

    # 1. Generate Mask
    mask_image = create_clothing_mask(foreground_image)

    # 2. Save to temporary bytes for Replicate API
    img_byte_arr = io.BytesIO()
    foreground_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    mask_byte_arr = io.BytesIO()
    mask_image.save(mask_byte_arr, format="PNG")
    mask_byte_arr.seek(0)

    # 3. Call Replicate Stable Diffusion Inpainting
    # Using a popular, production-ready inpainting model on Replicate
    output = replicate.run(
        "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
        input={
            "image": img_byte_arr,
            "mask": mask_byte_arr,
            "prompt": "A professional business suit, formal attire, photorealistic, tailored, high quality",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    )

    # 4. Download the generated image
    response = requests.get(output[0])
    suit_image = Image.open(io.BytesIO(response.content))
    return suit_image
