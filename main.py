import os
import cv2 as cv
import numpy as np

input_folder = 'fotos'
output_folder = 'fotos_canny'

os.makedirs(output_folder, exist_ok=True)

kernel = np.ones((3, 3), np.uint8)

# Supported image extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'canny_{filename}')

        img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Could not read image: {input_path}")
            continue

        edges = cv.Canny(img, 100, 200)

        edges_dilated = cv.dilate(edges, kernel, iterations=1)

        # Invert black and white colors of the Canny output
        edges_inverted = cv.bitwise_not(edges_dilated)

        cv.imwrite(output_path, edges_inverted)
        print(f"Saved: {output_path}")