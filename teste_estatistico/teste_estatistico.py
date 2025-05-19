import cv2
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import os

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.transpose())
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return float(similarity[0][0]) if isinstance(similarity, np.ndarray) else float(similarity)

def get_image_embedding(image, processor, model):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(1)
    return embedding.detach().numpy()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    original_img_path = os.path.join(project_root, "players", "enzo", "raposa.png")
    canny_img_path = os.path.join(project_root, "fotos_canny", "canny_raposa.png")
    
    original_img = cv2.imread(original_img_path)
    canny_img = cv2.imread(canny_img_path)
    
    if original_img is None:
        raise ValueError(f"Could not load original image from {original_img_path}")
    if canny_img is None:
        raise ValueError(f"Could not load canny image from {canny_img_path}")
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    canny_embedding = get_image_embedding(canny_img, processor, model)
    
    results = []
    
    for degree in range(0, 361, 1):
        
        for resize_percent in range(50, 151, 10):

            height, width = original_img.shape[:2]
            
            # Calculate new dimensions based on resize_percent
            new_width = int(width * resize_percent / 100)
            new_height = int(height * resize_percent / 100)
            
            # Resize image using OpenCV
            resized = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply rotation using OpenCV
            center = (new_width // 2, new_height // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

            rotated = cv2.warpAffine(resized, rotation_matrix, (new_width, new_height))
            
            # Get embedding for transformed image
            transformed_embedding = get_image_embedding(rotated, processor, model)
            
            # Calculate similarity
            similarity = calculate_cosine_similarity(canny_embedding, transformed_embedding)
            print(f"Degree: {degree}, Resize: {resize_percent}%, Similarity: {similarity:.4f}")
            
            results.append(f"{similarity:.4f}")
    
    output_path = os.path.join(current_dir, "transformation_results2.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    main()