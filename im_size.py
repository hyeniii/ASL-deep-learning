import os
from skimage import io

# set directory path
dir_path = "./data/test/images"

# initialize variables for total height and width
total_height = 0
total_width = 0

# count of images
count = 0

# loop through images in directory
for filename in os.listdir(dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # read image and get dimensions
        img_path = os.path.join(dir_path, filename)
        img = io.imread(img_path)
        height, width = img.shape[:2]
        
        # add to total height and width
        total_height += height
        total_width += width
        
        # increment count
        count += 1

# calculate average height and width
avg_height = total_height / count
avg_width = total_width / count

print("Average height:", avg_height)
print("Average width:", avg_width)
