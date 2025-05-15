from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np

url = "fotos/brasil.png"
url2 = "fotos_canny/canny_brasil.png"

# Open local images
image = Image.open(url).convert("RGB")
image2 = Image.open(url2).convert("RGB")


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

inputs = processor(images=image, return_tensors="pt")
inputs2 = processor(images=image2, return_tensors="pt")

outputs = model(**inputs)
outputs2 = model(**inputs2)

last_hidden_states = outputs.last_hidden_state[:, 0, :].squeeze(1)
last_hidden_states2 = outputs2.last_hidden_state[:, 0, :].squeeze(1)

def calculate_cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2.transpose())
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2)


print("Cosine Similarity:", calculate_cosine_similarity(last_hidden_states.detach().numpy(), last_hidden_states2.detach().numpy()))

