import os
import argparse
import xml.etree.ElementTree as ET
import json
import math

def convert_annotation(annotation_path, image_dir, category_id, annotations, images):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_filename = root.find("filename").text
    image_path = os.path.join(image_dir, image_filename)

    # Get the original image dimensions
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    # Resize the image to 224x224
    img_width_resized = 224
    img_height_resized = 224

    image_info = {
        "id": len(images) + 1,
        "width": img_width_resized,
        "height": img_height_resized,
        "file_name": image_filename,
    }
    images.append(image_info)

    for obj in root.findall("object"):
        category = obj.find("name").text
        if category != category_id:
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Calculate scaling factors for width and height
        scale_x = img_width_resized / img_width
        scale_y = img_height_resized / img_height

        # Scale down bounding box coordinates
        xmin_scaled = math.floor(xmin * scale_x)
        ymin_scaled = math.floor(ymin * scale_y)
        xmax_scaled = math.ceil(xmax * scale_x)
        ymax_scaled = math.ceil(ymax * scale_y)

        annotation_info = {
            "id": len(annotations) + 1,
            "image_id": image_info["id"],
            "category_id": 1,  # You can set the category_id as needed
            "bbox": [xmin_scaled, ymin_scaled, xmax_scaled - xmin_scaled, ymax_scaled - ymin_scaled],
            "area": (xmax_scaled - xmin_scaled) * (ymax_scaled - ymin_scaled),
            "iscrowd": 0,
            "ignore": 0,
        }
        annotations.append(annotation_info)
    
    # print(annotations)

    return annotations, images

def main(args):
    annotations = []
    images = []

    for filename in os.listdir(args.annotation_dir):
        if filename.endswith(".xml"):
            annotation_path = os.path.join(args.annotation_dir, filename)
            annotation, images = convert_annotation(annotation_path, args.image_dir, args.category, annotations, images)

        # print(annotations)

    output_data = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "fg", "supercategory": "fg"}],
        "images": images,
        "annotations": annotations,
    }

    with open(args.output_json, "w") as json_file:
        json.dump(output_data, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ground truth annotations to COCO format")
    parser.add_argument(
        "annotation_dir", type=str, help="Path to the directory containing ground truth annotations"
    )
    parser.add_argument(
        "image_dir", type=str, help="Path to the directory containing images"
    )
    parser.add_argument(
        "output_json", type=str, help="Path to the output JSON file"
    )
    parser.add_argument(
        "category", type=str, help="Category name in the ground truth annotations"
    )

    args = parser.parse_args()
    main(args)

