import os
import cv2
import json
import argparse
from tqdm import tqdm

def selective_search(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    # Perform selective search
    ss.switchToSelectiveSearchQuality()  # You can also use 'fast' for a faster but less accurate version
    rects = ss.process()

    return rects

def main(input_dir, output_json):
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    bbox_data = {}

    for image_file in tqdm(image_files):
        image_path = os.path.join(input_dir, image_file)

        # Perform selective search
        bboxes = selective_search(image_path)
        bbox_data[image_file] = bboxes.tolist()

    with open(output_json, "w") as json_file:
        json.dump(bbox_data, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform selective search on images and save bounding boxes as JSON.")
    parser.add_argument("input_directory", type=str, help="Path to the input directory containing images")
    parser.add_argument("output_json_file", type=str, help="Path to the output JSON file for bounding boxes")

    args = parser.parse_args()
    main(args.input_directory, args.output_json_file)
