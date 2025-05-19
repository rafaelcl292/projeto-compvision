import os
import cv2 as cv
import numpy as np

input_folder = 'players/marcelo'
output_folder = 'fotos_dilate_variations'
filename = 'mack.png'  # Specific file to process

os.makedirs(output_folder, exist_ok=True)

kernel = np.ones((3, 3), np.uint8)

input_path = os.path.join(input_folder, filename)

# Read the image
img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print(f"Could not read image: {input_path}")
    exit()

# Apply Canny edge detection
edges = cv.Canny(img, 100, 200)

# Generate 5 erosion variations
for i in range(1, 6):
    # Apply erosion with current iteration count
    edges_eroded = cv.erode(edges, kernel, iterations=i)
    
    # Invert black and white colors
    edges_inverted = cv.bitwise_not(edges_eroded)
    
    # Create output filename with iteration count
    output_filename = f'erode_{i}_{filename}'
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the image
    cv.imwrite(output_path, edges_inverted)
    print(f"Saved erosion: {output_path}")

# Generate 5 dilation variations
for i in range(1, 6):
    # Apply dilation with current iteration count
    edges_dilated = cv.dilate(edges, kernel, iterations=i)
    
    # Invert black and white colors
    edges_inverted = cv.bitwise_not(edges_dilated)
    
    # Create output filename with iteration count
    output_filename = f'dilate_{i}_{filename}'
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the image
    cv.imwrite(output_path, edges_inverted)
    print(f"Saved dilation: {output_path}") 