import os
import json
import argparse
import csv
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

def calculate_iou(box1, box2):
    # Calculate intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of boxes
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    # Calculate IoU
    iou = intersection / (area_box1 + area_box2 - intersection)
    return iou

def evaluate_image(gt_boxes, dt_boxes, iou_thresholds, image_path, draw_directory=None):
    results = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in iou_thresholds}
    recall_results = []

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for gt_box in gt_boxes:
        for dt_box in dt_boxes:
            iou = calculate_iou(gt_box, dt_box)

            for threshold in iou_thresholds:
                if iou >= threshold and results[threshold]["tp"] == 0:
                    results[threshold]["tp"] = 1
                else:
                    results[threshold]["fp"] += 1

    for threshold in iou_thresholds:
        results[threshold]["fn"] = len(gt_boxes) - results[threshold]["tp"]

    for threshold in iou_thresholds:
        recall = results[threshold]["tp"] / (results[threshold]["tp"] + results[threshold]["fn"])
        recall_results.append(recall)

    if draw_directory:
        image_with_boxes = image.copy()
        draw_with_boxes = ImageDraw.Draw(image_with_boxes)

        for box in gt_boxes:
            x, y, w, h = box
            draw_with_boxes.rectangle([x, y, x + w, y + h], outline="green")
            draw_with_boxes.text((x, y), "Ground Truth", fill="green", font=font)

        for box in dt_boxes:
            x, y, w, h = box
            draw_with_boxes.rectangle([x, y, x + w, y + h], outline="red")
            draw_with_boxes.text((x, y), "Detection", fill="red", font=font)

        image_with_boxes.save(os.path.join(draw_directory, os.path.basename(image_path)))

    return results, recall_results

def main(args):
    gt_dir = args.gt_directory
    image_dir = args.image_directory
    detection_file = args.detection_file
    iou_thresholds = [0.15, 0.3, 0.5, 0.75, 0.85, 0.9]

    # Load detection data
    with open(detection_file, "r") as dt_file:
        detection_data = json.load(dt_file)

    image_info = detection_data['images']
    annotations = detection_data["annotations"]
    name_to_id_mapper = {}
    id_to_ann_mapper = {}

    for info in image_info:
        name = info["file_name"][2:]
        im_id = info["id"]
        name_to_id_mapper.update({name: im_id})
        
    for ann in annotations:
        im_id = ann["image_id"]
        ann_list = []
        if im_id in id_to_ann_mapper:
            ann_list = id_to_ann_mapper[im_id]
        ann_list.append(ann)
        id_to_ann_mapper.update({im_id: ann_list})

    mAP_sum = 0
    recall_sum = {threshold: 0 for threshold in iou_thresholds}

    for threshold in iou_thresholds:
        precision_sum = 0
        image_count = 0
        skipped_images = 0

        for image_filename in os.listdir(image_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                skipped_images += 1
                continue  # Skip non-image files

            if image_filename not in name_to_id_mapper:
                continue

            gt_boxes = []
            gt_annotation_filename = f"{gt_dir}{image_filename[:-4]}.xml"
           
            if not os.path.exists(gt_annotation_filename):
                skipped_images += 1
                continue

            gt_ann = ET.parse(gt_annotation_filename)
            gt_ann_root = gt_ann.getroot()
            im_width = int(gt_ann_root.find("size/width").text)
            im_height = int(gt_ann_root.find("size/height").text)
            
            bbox = gt_ann_root.find("object/bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            scale_x = 224 / im_width
            scale_y = 224 / im_height

            xmin = math.floor(xmin * scale_x)
            ymin = math.floor(ymin * scale_y)
            xmax = math.ceil(xmax * scale_x)
            ymax = math.ceil(ymax * scale_y)
            
            gt_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

            if image_filename in name_to_id_mapper:
                im_id = name_to_id_mapper[image_filename]
                ann_list = id_to_ann_mapper[im_id]
                dt_boxes = [ann["bbox"] for ann in ann_list]
            else:
                dt_boxes = []  # No detections for this image

            results, recall_results = evaluate_image(gt_boxes, dt_boxes, [threshold], os.path.join(image_dir, image_filename), args.draw_directory)
            precision = results[threshold]["tp"] / (results[threshold]["tp"] + results[threshold]["fp"])
            precision_sum += precision
            recall_sum[threshold] += sum(recall_results)
            image_count += 1

        print(f"Skipped {skipped_images} non-image files.")
        print(f"Processed {image_count} images.")
        average_precision = precision_sum / image_count
        mAP_sum += average_precision

        print(f"IoU Threshold {threshold}:")
        print(f"Average Precision: {average_precision}")
        print(f"Average Recall: {recall_sum[threshold] / image_count}")

    final_mAP = mAP_sum / len(iou_thresholds)
    final_recall = {threshold: recall_sum[threshold] / image_count for threshold in iou_thresholds}

    print(f"Mean Average Precision (mAP): {final_mAP}")
    print("Average Recall:", final_recall)

    # Save results to a CSV file
    output_file = args.output_csv
    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['IoU Threshold', 'Average Precision', 'Average Recall']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for threshold in iou_thresholds:
            writer.writerow({'IoU Threshold': threshold, 'Average Precision': final_mAP, 'Average Recall': final_recall[threshold]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection results")
    parser.add_argument(
        "gt_directory", type=str, help="Directory containing ground truth annotations in XML format"
    )
    parser.add_argument(
        "image_directory", type=str, help="Directory containing ground truth images"
    )
    parser.add_argument(
        "detection_file", type=str, help="Detection annotations in JSON format"
    )
    parser.add_argument(
        "output_csv", type=str, help="Path to save the results as a CSV file"
    )
    parser.add_argument(
        "--draw_directory", type=str, default=None, help="Directory to save images with labeled bounding boxes (optional)"
    )

    args = parser.parse_args()
    main(args)
