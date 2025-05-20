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


def get_all_available_images():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    players_dir = os.path.join(project_root, "players")

    available_images = []
    for player_name in os.listdir(players_dir):
        player_dir = os.path.join(players_dir, player_name)
        if os.path.isdir(player_dir):
            for image_file in os.listdir(player_dir):
                if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    canny_path = os.path.join(
                        project_root, "fotos_canny", f"canny_{image_file}"
                    )
                    if os.path.exists(canny_path):
                        available_images.append((player_name, image_file))
    return available_images


def ensure_results_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "transformation_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def process_single_image(player_name, image_file, processor, model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    player_dir = os.path.join(project_root, "players", player_name)
    canny_dir = os.path.join(project_root, "fotos_canny")

    print(f"\nProcessing {player_name}/{image_file}...")

    original_img_path = os.path.join(player_dir, image_file)
    canny_img_path = os.path.join(canny_dir, f"canny_{image_file}")

    if not os.path.exists(canny_img_path):
        print(f"Canny image not found for {image_file}, skipping...")
        return

    original_img = cv2.imread(original_img_path)
    canny_img = cv2.imread(canny_img_path)

    if original_img is None or canny_img is None:
        print(f"Failed to load images for {image_file}, skipping...")
        return

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
                transformed_embedding = get_image_embedding(dilated, processor, model)

                # Calculate similarity
                similarity = calculate_cosine_similarity(
                    canny_embedding, transformed_embedding
                )
                print(f"{similarity:.4f}")

                results.append(f"{similarity}")

    # Save results for this image in the transformation_results directory
    results_dir = ensure_results_directory()
    output_filename = (
        f"transformation_results_{player_name}_{os.path.splitext(image_file)[0]}.txt"
    )
    output_path = os.path.join(results_dir, output_filename)
    with open(output_path, "w") as f:
        f.write("\n".join(results))
    print(f"\nResults saved to {output_filename} in transformation_results directory")


def main():
    # Initialize the model and processor once
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Ensure the results directory exists
    ensure_results_directory()

    while True:
        # Get all available images
        available_images = get_all_available_images()

        if not available_images:
            print("No images found with corresponding Canny versions!")
            break

        print("\nAvailable images:")
        for idx, (player_name, image_file) in enumerate(available_images, 1):
            print(f"{idx}. {player_name}/{image_file}")
        print("0. Exit")

        try:
            choice = input(
                "\nEnter the number of the image to process (or 0 to exit): "
            )

            if choice == "0":
                break

            choice = int(choice)
            if 1 <= choice <= len(available_images):
                player_name, image_file = available_images[choice - 1]
                process_single_image(player_name, image_file, processor, model)
            else:
                print("Invalid choice! Please try again.")
        except ValueError:
            print("Please enter a valid number!")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
