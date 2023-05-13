import os

import cv2
import pandas as pd

def process_images_indirect(input_dir, output_dir, annotations_csv):
    bbox_info = pd.read_csv(annotations_csv)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for label in os.listdir(input_dir):
        input_label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
        
        for image_name in os.listdir(input_label_dir):
            # Check if the file extension is .jpg
            if image_name.lower().endswith(".jpg"):
                image_path = os.path.join(input_label_dir, image_name)
                img = cv2.imread(image_path)

                # Get the bounding box information for this image
                bbox_rows = bbox_info[bbox_info["filename"] == image_name]

                if not bbox_rows.empty:
                    bbox = bbox_rows.iloc[0]
                    xmin, xmax, ymin, ymax = bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]

                    # Crop the hand region
                    cropped_img = img[ymin:ymax, xmin:xmax]

                    # Save the cropped image to the output directory
                    output_image_path = os.path.join(output_label_dir, image_name)
                    cv2.imwrite(output_image_path, cropped_img)
                else:
                    print(f"No bounding box information found for {image_name}. Skipping this image.")
            else:
                print(f"Skipping non-jpg file {image_name}")
                

def process_images_direct(input_dir, output_dir, annotations_csv):
    bbox_info = pd.read_csv(annotations_csv)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(input_dir):
        # Check if the file extension is '.jpg'
        if image_name.lower().endswith('.jpg'):
            image_path = os.path.join(input_dir, image_name)
            img = cv2.imread(image_path)

            # Get the bounding box information for this image
            bbox_rows = bbox_info[bbox_info['filename'] == image_name]

            if not bbox_rows.empty:
                bbox = bbox_rows.iloc[0]
                xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']

                # Crop the hand region
                cropped_img = img[ymin:ymax, xmin:xmax]

                # Save the cropped image to the output directory
                output_image_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_image_path, cropped_img)
            else:
                print(f"No bounding box information found for {image_name}. Skipping this image.")
        else:
            print(f"Skipping non-jpg file {image_name}")
