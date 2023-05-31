import os
import shutil

def restructure_dataset(dataset_dir, output_dir):
    # Iterate through each image file in the dataset directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            # Get the file path
            file_path = os.path.join(dataset_dir, filename)

            # Split the filename using the first _ as the delimiter
            split_result = filename.split('_', 1)

            # Retrieve the class label from the split result
            class_label = split_result[0][0]

            # Create a folder for the class if it doesn't exist
            class_folder = os.path.join(output_dir, class_label)

            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Move the image file to the corresponding class folder
            shutil.move(file_path, os.path.join(class_folder, filename))