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
    return (
        float(similarity[0][0])
        if isinstance(similarity, np.ndarray)
        else float(similarity)
    )


def get_image_embedding(image, processor, model):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(1)
    return embedding.detach().numpy()


def process_player_images(player_name, processor, model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    player_dir = os.path.join(project_root, "players", player_name)
    canny_dir = os.path.join(project_root, "fotos_canny")

    # Get all image files from player directory
    image_files = [
        f
        for f in os.listdir(player_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        print(f"Processing {player_name}/{image_file}...")

        original_img_path = os.path.join(player_dir, image_file)
        canny_img_path = os.path.join(canny_dir, f"canny_{image_file}")

        if not os.path.exists(canny_img_path):
            print(f"Canny image not found for {image_file}, skipping...")
            continue

        original_img = cv2.imread(original_img_path)
        canny_img = cv2.imread(canny_img_path)

        if original_img is None or canny_img is None:
            print(f"Failed to load images for {image_file}, skipping...")
            continue

        canny_embedding = get_image_embedding(canny_img, processor, model)
        results = []

        kernel = np.ones((3, 3), np.uint8)

        # Test all combinations of rotation, resize and dilation
        for degree in range(0, 361, 18):
            for resize_percent in range(50, 151, 5):
                for dilation_iter in range(1, 4):
                    height, width = original_img.shape[:2]

                    # Calculate new dimensions based on resize_percent
                    new_width = int(width * resize_percent / 100)
                    new_height = int(height * resize_percent / 100)

                    # Resize image using OpenCV
                    resized = cv2.resize(
                        original_img,
                        (new_width, new_height),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                    # Apply rotation using OpenCV
                    center = (new_width // 2, new_height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
                    rotated = cv2.warpAffine(
                        resized, rotation_matrix, (new_width, new_height)
                    )

                    # Apply dilation
                    dilated = cv2.dilate(rotated, kernel, iterations=dilation_iter)

                    # Get embedding for transformed image
                    transformed_embedding = get_image_embedding(
                        dilated, processor, model
                    )

                    # Calculate similarity
                    similarity = calculate_cosine_similarity(
                        canny_embedding, transformed_embedding
                    )
                    print(f"{similarity:.4f}")

                    results.append(f"{similarity}")

        # Save results for this image
        output_filename = f"transformation_results_{player_name}_{os.path.splitext(image_file)[0]}.txt"
        output_path = os.path.join(current_dir, output_filename)
        with open(output_path, "w") as f:
            f.write("\n".join(results))


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    players_dir = os.path.join(project_root, "players")

    # Get all player directories
    player_names = [
        d
        for d in os.listdir(players_dir)
        if os.path.isdir(os.path.join(players_dir, d))
    ]

    # Initialize the model and processor once
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Process each player's images
    for player_name in player_names:
        print(f"\nProcessing player: {player_name}")
        process_player_images(player_name, processor, model)


if __name__ == "__main__":
    main()
