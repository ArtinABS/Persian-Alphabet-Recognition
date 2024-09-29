
import os
import cv2
import random
import numpy as np
def process_and_merge_images(folder1, folder2, output_folder, output_size=(50, 50), white_threshold=0.8):
    def process_image(image_path, output_size, white_threshold):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, output_size)
        total_pixels = resized_image.size
        white_pixels = np.sum(resized_image > 240)
        white_ratio = white_pixels / total_pixels

        if white_ratio >= white_threshold:
            inverted_color_image = cv2.bitwise_not(resized_image)
            return inverted_color_image
        else:
            return resized_image

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = []

    for root, dirs, files in os.walk(folder1):
        for file in files:
            if file.endswith(('.jpg')):
                image_paths.append(os.path.join(root, file))

    for root, dirs, files in os.walk(folder2):
        for file in files:
            if file.endswith(('.jpg')):
                image_paths.append(os.path.join(root, file))
                
    random.shuffle(image_paths)

    for image_path in image_paths:
        processed_image = process_image(image_path, output_size, white_threshold)
        file_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_image_path, processed_image)
        print(f"Processed and saved: {output_image_path}")

folder1 = r'C:\Users\NoteBook\Desktop\alphabet\Datasets\DS-1'
folder2 = r'C:\Users\NoteBook\Desktop\alphabet\Datasets\DS-2'
output_folder = r'C:\Users\NoteBook\Desktop\alphabet\Datasets\preprosseced all data'

process_and_merge_images(folder1, folder2, output_folder, output_size=(50, 50), white_threshold=0.8)