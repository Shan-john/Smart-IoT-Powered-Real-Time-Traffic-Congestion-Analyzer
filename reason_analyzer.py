# reason_analyzer.py
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define congestion reasons
REASONS = [
    "normal traffic",
    "heavy traffic",
    "wrong parking",
    "road blocked",
    "accident on the road",
    "vehicles moving slowly",
    "frequent braking",
    "vehicles stopped without reason",
    "traffic signal not working"
]

def analyze_congestion_reason(image_pil: Image.Image) -> str:
    inputs = clip_processor(text=REASONS, images=image_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
    return REASONS[best_idx]
