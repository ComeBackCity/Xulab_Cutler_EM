import os
import json
import argparse
import csv
import math
import xml.etree.ElementTree as ET

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

def evaluate_image(gt_boxes, dt_boxes, iou_thresholds):
    results = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in iou_thresholds}
    min_ious = {threshold: threshold - 1e-6 for threshold in iou_thresholds}

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

    return results

def main(args):
    gt_dir = args.gt_directory
    image_dir = args.image_directory
    detection_file = args.detection_file
    iou_thresholds = [0.15, 0.3, 0.5, 0.75, 0.85, 0.9]

    with open(detection_file, "r") as dt_file:
        detection_data = json.load(dt_file)

    mAP_sum = 0
    APs = {threshold: 0 for threshold in iou_thresholds}
    ARs = {threshold: 0 for threshold in iou_thresholds}

    for threshold in iou_thresholds:
        precision_sum = 0
        recall_sum = 0
        image_count = 0
        skipped_images = 0

        for image_filename in os.listdir(image_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                skipped_images += 1
                continue  # Skip non-image files

            if image_filename not in detection_data:
                continue

            gt_boxes = []  # You need to load the corresponding ground truth boxes
            gt_annotation_filename = f"{gt_dir}{image_filename[:-4]}.xml"
           
            if not os.path.exists(gt_annotation_filename):
                skipped_images += 1
                continue

            gt_ann = ET.parse(gt_annotation_filename)
            gt_ann_root = gt_ann.getroot()
            # print(gt_ann)
            im_width = int(gt_ann_root.find("size/width").text)
            im_height = int(gt_ann_root.find("size/height").text)
            
            bbox = gt_ann_root.find("object/bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Calculate scaling factors for width and height
            scale_x = 224 / im_width
            scale_y = 224 / im_height

            # Scale down bounding box coordinates
            xmin = math.floor(xmin * scale_x)
            ymin = math.floor(ymin * scale_y)
            xmax = math.ceil(xmax * scale_x)
            ymax = math.ceil(ymax * scale_y)
            
            gt_boxes.append([xmin, ymin, xmax-xmin, ymax-ymin])
    
            if image_filename in detection_data:
                dt_boxes = detection_data[image_filename]
                # print(len(dt_boxes))
            else:
                dt_boxes = []  # No detections for this image

            results = evaluate_image(gt_boxes, dt_boxes, [threshold])
            precision = results[threshold]["tp"] / (results[threshold]["tp"] + results[threshold]["fp"])
            recall = results[threshold]["tp"] / (results[threshold]["tp"] + results[threshold]["fn"])
            precision_sum += precision
            recall_sum += recall
            image_count += 1
        
        # print(skipped_images)
        # print(image_count)
        average_precision = precision_sum / image_count
        APs[threshold] = average_precision
        average_recall = recall_sum / image_count
        ARs[threshold] = average_recall

        print(f"IoU Threshold {threshold}:")
        print(f"Average Precision: {average_precision}")
        print(f"Average Recall: {average_recall}")

    # Save results to a CSV file
    output_file = args.output_csv
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            pass

    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['IoU Threshold', 'Average Precision', 'Average Recall']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for threshold in iou_thresholds:
            writer.writerow({'IoU Threshold': threshold, 'Average Precision': APs[threshold], 'Average Recall': ARs[threshold]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection results")
    parser.add_argument(
        "gt_directory", type=str, help="Directory containing ground truth annotations in JSON format"
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

    args = parser.parse_args()
    main(args)

