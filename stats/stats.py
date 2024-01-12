import os
import cv2
import numpy as np
import sys

def calculate_mean_std(directory):
    # Initialize variables to store cumulative sums
    total_mean = 0.0
    total_std = 0.0
    num_images = 0

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Read the image
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Convert pixel values to float
            img_float = img.astype(np.float32)

            # Update cumulative sums
            total_mean += np.mean(img_float)
            total_std += np.std(img_float)
            num_images += 1

    # Calculate the mean and standard deviation
    overall_mean = total_mean / num_images
    overall_std = total_std / num_images

    return overall_mean, overall_std

# Example usage
directory_path = sys.argv[1]
mean, std = calculate_mean_std(directory_path)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

