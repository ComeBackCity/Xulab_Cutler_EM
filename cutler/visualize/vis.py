import os
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
import random

def visualize_and_save_annotations(annotation_file, image_dir, output_dir):
    # Load COCO format annotations
    with open(annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    # print(coco_annotations)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save each image
    for image_info in coco_annotations['images']:
        # Load the RGB image using Pillow
        image_path = os.path.join(image_dir, image_info['file_name'])
        # print(image_path)
        image = Image.open(image_path).convert('RGB')

        # Retrieve annotations for the current image
        annotations = [ann for ann in coco_annotations['annotations'] if ann['image_id'] == image_info['id']]

        # Create a drawing object for annotations


        # Assign a unique color to each mask while keeping the same color for the mask and its respective bounding box
        for idx, annotation in enumerate(annotations):
            mask_color = tuple(random.sample(range(0, 255), 3))

            # Extract segmentation mask
            segmentation = annotation['segmentation']  # Assuming only one segmentation per annotation

            # Convert segmentation to a binary mask
            mask = mask_utils.decode(segmentation)

            # Create an RGBA image with object color splash
            splash_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            splash_image.paste(mask_color + (128,), mask=Image.fromarray((mask * 255).astype(np.uint8)))

            # Composite the splash image onto the original image
            image = Image.alpha_composite(image.convert('RGBA'), splash_image)
            draw = ImageDraw.Draw(image)

            # Draw bounding box
            bbox = annotation['bbox']
            x, y, w, h = bbox
            # print(bbox)
            # print(mask_color)
            draw.rectangle([x, y, x + w, y + h], outline=mask_color, width=2)

        # Save the annotated image to the output directory
        output_image_path = os.path.join(output_dir, image_info['file_name'])
        image = image.convert('RGB')  # Convert back to RGB mode
        image.save(output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and save images with COCO format annotations')
    parser.add_argument('annotation_file', type=str, help='Path to the COCO format annotation file')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing images')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for annotated images')

    args = parser.parse_args()

    visualize_and_save_annotations(args.annotation_file, args.image_dir, args.output_dir)

