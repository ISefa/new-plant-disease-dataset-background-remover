import os
from rembg import remove
from PIL import Image
import cv2
import numpy as np
from PIL import Image

# data paths
DATA_DIR_TRAIN = r""  # train source
DATA_DIR_VALID = r""  # valid source
OUTPUT_DIR_TRAIN = r""  # train target
OUTPUT_DIR_VALID = r""  # valid target

def remove_background(input_dir, output_dir, completed_classes):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        if class_name in completed_classes:
            print(f"{class_name} class has already been completed, skipping...")
            continue

        input_class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        if class_name == "Corn":
            print(f"{class_name} skipping corn class")
            for img_name in os.listdir(input_class_path):
                img_path = os.path.join(input_class_path, img_name)
                output_path = os.path.join(output_class_path, img_name)
                Image.open(img_path).save(output_path)
            continue

        print(f"{class_name} Removing background for class...")
        for img_name in os.listdir(input_class_path):
            print(f"{img_name} processing")
            img_path = os.path.join(input_class_path, img_name)
            output_path = os.path.join(output_class_path, img_name)

            with open(img_path, "rb") as img_file:
                input_image = img_file.read()
                output_image = remove(input_image)

                input_array = np.frombuffer(input_image, dtype=np.uint8)
                output_array = np.frombuffer(output_image, dtype=np.uint8)

                original_image = cv2.imdecode(input_array, cv2.IMREAD_COLOR)
                rembg_result = cv2.imdecode(output_array, cv2.IMREAD_COLOR)

                fixed_image = preserve_green_regions(original_image, rembg_result)

                fixed_image_pil = Image.fromarray(cv2.cvtColor(fixed_image, cv2.COLOR_BGR2RGB))
                fixed_image_pil.save(output_path)
                print(f"{output_path} completed ({fixed_image.__len__()})")

    print(f"Background removal has been completed for {output_dir}")

def preserve_green_regions(original_image, rembg_result):
    # Convert original image to HSV
    hsv_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Green color ranges (HSV values)
    lower_green = np.array([35, 40, 40])  # Lower bound (hue, saturation, value)
    upper_green = np.array([85, 255, 255])  # Upper bound (hue, saturation, value)

    # Mask green areas
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Combine original and background-removed image
    combined_result = np.where(green_mask[:, :, None] > 0, original_image, rembg_result)

    return combined_result

if __name__ == "__main__":
    print("Starting background removal process...")

    # Previously completed classes
    completed_train_classes = [""]  # Train part not started yet. This area for already complated classes for skipping.
    completed_valid_classes = [""]  # Validation part not started yet. This area for already complated classes for skipping.

    # Background removal process for train and validation datasets
    remove_background(DATA_DIR_TRAIN, OUTPUT_DIR_TRAIN, completed_train_classes)
    remove_background(DATA_DIR_VALID, OUTPUT_DIR_VALID, completed_valid_classes)

    print("Background removal process completed.")
