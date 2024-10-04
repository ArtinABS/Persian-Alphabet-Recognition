import os
import cv2
import random
import numpy as np

def process_image(image_path, output_size, white_threshold):
    """Process a single image by resizing and inverting if necessary."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, output_size)

    # Calculate white pixel ratio
    total_pixels = resized_image.size
    white_ratio = np.sum(resized_image > 240) / total_pixels

    # Invert color if the white pixel ratio is above the threshold
    return cv2.bitwise_not(resized_image) if white_ratio >= white_threshold else resized_image

def load_image_paths(dataset):
    """Load all image paths from the given dataset directory."""
    image_paths = []
    for root, _, files in os.walk(dataset):
        image_paths.extend(os.path.join(root, file) for file in files if file.endswith('.jpg'))
    return image_paths

def process_and_merge_images(dataset1, dataset2, output_dataset, output_size=(50, 50), white_threshold=0.8):
    """Process images from two datasets and save the results in the output directory."""
    if not os.path.exists(output_dataset):
        os.makedirs(output_dataset)

    image_paths = load_image_paths(dataset1) + load_image_paths(dataset2)
    random.shuffle(image_paths)

    for image_path in image_paths:
        processed_image = process_image(image_path, output_size, white_threshold)
        file_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_dataset, file_name)
        cv2.imwrite(output_image_path, processed_image)
        print(f"Processed and saved: {output_image_path}")

# Define dataset paths
dataset1 = r'C:\Users\NoteBook\Desktop\alphabet\Datasets\DS-1'
dataset2 = r'C:\Users\NoteBook\Desktop\alphabet\Datasets\DS-2'
output_dataset = r'C:\Users\NoteBook\Desktop\alphabet\Datasets\preprocessed_all_data'

# Run the processing and merging
process_and_merge_images(dataset1, dataset2, output_dataset)
