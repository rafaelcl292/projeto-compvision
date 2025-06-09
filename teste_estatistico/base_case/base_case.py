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

def ensure_results_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "transformation_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def generate_variations(image, num_variations=1000):
    variations = []
    kernel = np.ones((3, 3), np.uint8)
    
    # Calculate step sizes to get approximately 1000 variations
    # We'll use 10 rotation steps, 10 resize steps, and 10 dilation steps
    rotation_steps = np.linspace(0, 360, 10)
    resize_steps = np.linspace(50, 150, 10)
    dilation_steps = range(1, 4)
    
    for degree in rotation_steps:
        for resize_percent in resize_steps:
            for dilation_iter in dilation_steps:
                height, width = image.shape[:2]
                
                # Calculate new dimensions based on resize_percent
                new_width = int(width * resize_percent / 100)
                new_height = int(height * resize_percent / 100)
                
                # Resize image
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Apply rotation
                center = (new_width // 2, new_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
                rotated = cv2.warpAffine(resized, rotation_matrix, (new_width, new_height))
                
                # Apply dilation
                dilated = cv2.dilate(rotated, kernel, iterations=dilation_iter)
                
                variations.append(dilated)
                
                if len(variations) >= num_variations:
                    return variations
    
    return variations

def process_base_case():
    # Initialize the model and processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Get current directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Paths for insper.png and canny directory
    insper_path = "insper.png"
    canny_dir = os.path.join(project_root, "fotos_canny")

    if not os.path.exists(insper_path):
        print("insper.png not found!")
        return

    # Load insper.png
    insper_img = cv2.imread(insper_path)
    if insper_img is None:
        print("Failed to load insper.png!")
        return

    # Generate variations
    print("Generating variations...")
    variations = generate_variations(insper_img)
    print(f"Generated {len(variations)} variations")

    # Get embeddings for all variations
    print("Calculating embeddings for variations...")
    variation_embeddings = [get_image_embedding(var, processor, model) for var in variations]

    # Create results directory
    results_dir = ensure_results_directory()

    # Process each Canny image
    for canny_file in os.listdir(canny_dir):
        if canny_file.lower().endswith((".png", ".jpg", ".jpeg")):
            canny_path = os.path.join(canny_dir, canny_file)
            canny_img = cv2.imread(canny_path)
            
            if canny_img is None:
                print(f"Failed to load {canny_file}, skipping...")
                continue

            print(f"\nProcessing {canny_file}...")
            
            # Get Canny image embedding
            canny_embedding = get_image_embedding(canny_img, processor, model)

            # Calculate similarities for all variations
            similarities = []
            for i, var_embedding in enumerate(variation_embeddings):
                similarity = calculate_cosine_similarity(canny_embedding, var_embedding)
                similarities.append(similarity)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} variations...")

 
            
            # Create a separate file for each Canny image
            base_filename = os.path.splitext(canny_file)[0]
            output_file = os.path.join(results_dir, f"similarities_{base_filename}.txt")
            
            with open(output_file, "w") as f:
            
                f.write("\n".join([f"{s:.4f}" for s in similarities]))
            
            print(f"Results saved to {output_file}")

if __name__ == "__main__":
    process_base_case()
