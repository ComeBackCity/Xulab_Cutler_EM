import os
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

def visualize_and_save_annotations(annotation_file, image_dir, output_dir):
    # Load COCO format annotations
    with open(annotation_file, 'r') as f:
        coco_annotations = json.load(f)
    
    print(coco_annotations)
    # Create a color map for object categories
    category_colors = {}
    for category in coco_annotations['categories']:
        category_colors[category['id']] = tuple(int(255 * x) for x in np.random.rand(3))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save each image
    for image_info in coco_annotations['images']:
        # Load the RGB image using Pillow
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = Image.open(image_path)

        # Retrieve annotations for the current image
        annotations = [ann for ann in coco_annotations['annotations'] if ann['image_id'] == image_info['id']]

        # Create a drawing object for annotations
        draw = ImageDraw.Draw(image)

        # Add segmentation masks and object color splashes for annotations
        for annotation in annotations:
            category_id = annotation['category_id']
            color = category_colors.get(category_id, (255, 0, 0))
            
            print(annotation['segmentation'])
            # Extract segmentation mask
            segmentation = annotation['segmentation'][0]  # Assuming only one segmentation per annotation

            # Convert segmentation to a binary mask
            mask = mask_utils.decode(segmentation)

            # Create a mask image using Pillow
            mask_image = Image.fromarray(mask * 255).convert('L')

            # Create an RGBA image with object color splash
            splash_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            splash_image.paste(color + (128,), mask=mask_image)

            # Composite the splash image onto the original image
            image = Image.alpha_composite(image.convert('RGBA'), splash_image)

        # Save the annotated image to the output directory
        output_image_path = os.path.join(output_dir, image_info['file_name'])
        image.convert('RGB').save(output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and save images with COCO format annotations')
    parser.add_argument('annotation_file', type=str, help='Path to the COCO format annotation file')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing images')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for annotated images')

    args = parser.parse_args()

    visualize_and_save_annotations(args.annotation_file, args.image_dir, args.output_dir)

