import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# Define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_image_and_labels(image, bboxes, class_labels):
    """
    Apply augmentations to an image and its bounding boxes.

    Args:
        image: Input image (numpy array).
        bboxes: List of bounding boxes in YOLO format.
        class_labels: List of class labels.

    Returns:
        Tuple of augmented image and updated bounding boxes.
    """
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    return augmented_image, augmented_bboxes

def save_augmented_image_and_label(augmented_image, augmented_bboxes, image_output_path, label_output_path, class_labels):
    """
    Save augmented image and corresponding YOLO-format labels.

    Args:
        image: Augmented image.
        bboxes: Updated bounding boxes.
        img_path: Path to save the augmented image.
        label_path: Path to save YOLO labels.
        class_labels: List of class labels.
    """
    cv2.imwrite(image_output_path, augmented_image)

    # Save the updated label file in YOLO format (relative coordinates)
    with open(label_output_path, 'w') as f:
        for bbox, class_label in zip(augmented_bboxes, class_labels):
            # Each label is in the format: class_id x_center y_center width height (relative to image dimensions)
            x_center, y_center, width, height = bbox
            f.write(f"{class_label} {x_center} {y_center} {width} {height}\n")

def augment_images_in_directory(input_img_dir, input_label_dir, output_img_dir, output_label_dir):
    """
    Apply augmentations to all images in the input directory.

    Args:
        input_img_dir: Path to the original images.
        input_label_dir: Path to the corresponding YOLO labels.
        output_img_dir: Path to save augmented images.
        output_label_dir: Path to save augmented labels.
    """
    Path(output_img_dir).mkdir(parents=True, exist_ok=True)
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(os.listdir(input_img_dir)):
        img_path = os.path.join(input_img_dir, img_file)
        label_path = os.path.join(input_label_dir, img_file.replace(".jpg", ".txt"))

        # Read the image and corresponding labels
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Unable to read {img_path}")
            continue

        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        bboxes = []
        class_labels = []

        for line in label_lines:
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)

        # Apply augmentation
        augmented_image, augmented_bboxes = augment_image_and_labels(image, bboxes, class_labels)

        # Define output paths
        output_image_path = os.path.join(output_img_dir, img_file)
        output_label_path = os.path.join(output_label_dir, img_file.replace(".jpg", ".txt"))

        # Save augmented image and label
        save_augmented_image_and_label(augmented_image, augmented_bboxes, output_image_path, output_label_path, class_labels)

# Run augmentation
augment_images_in_directory(
    input_img_dir="drive/MyDrive/archive (1)/VehiclesDetectionDataset/train/images",  # Your original images directory
    input_label_dir="drive/MyDrive/archive (1)/VehiclesDetectionDataset/train/labels",  # Your original label directory (YOLO format)
    output_img_dir="drive/MyDrive/archive (1)/VehiclesDetectionDataset/augmented_train/images",  # Directory to save augmented images
    output_label_dir="drive/MyDrive/archive (1)/VehiclesDetectionDataset/augmented_train/labels"  # Directory to save augmented labels
)