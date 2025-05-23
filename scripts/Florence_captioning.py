"""
Demo: Using Florence-2 for Image Captioning

This script demonstrates how to use Microsoft's Florence-2 model to generate detailed captions for images.
It loads a sample image from the CIFAR-10 dataset, runs Florence-2, and prints the generated caption.

"""
# ------------------- Imports and Florence Setup -------------------
import os
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Florence-2 model and processor (no token needed for public models)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

# ------------------- Load a Sample Image -------------------
# Load CIFAR-10 dataset and select the first image
cifar_dataset = load_dataset("cifar10", split="train")
image_array = cifar_dataset["img"][0]  # Get the first image (as a NumPy array)
img = Image.fromarray(image_array)

# ------------------- Florence Caption Generation -------------------
def florence(image):
    """Generate a detailed caption using Florence-2."""
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,  # Shorter for demo
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    return list(parsed_answer.values())[0]

# ------------------- Run Florence and Print Caption -------------------
caption = florence(img)
print("Generated Florence Caption:")
print(caption)
