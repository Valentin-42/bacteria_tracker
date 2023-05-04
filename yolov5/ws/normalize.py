import os
import cv2
import numpy as np

# Define the path to the folder containing the images
folder_path = "~/local_storage/bacteria_tracker/datasets/500IMGS/images/train"

# Define the path to the folder for the normalized images
normalized_folder_path = "~/local_storage/bacteria_tracker/datasets/500IMGS/images/normalized"

# Create the normalized folder if it doesn't already exist
if not os.path.exists(normalized_folder_path):
    os.makedirs(normalized_folder_path)

# Loop over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(folder_path, filename))

        # Convert to float32 data type
        img = img.astype(np.float32)

        # Normalize pixel values between 0 and 1
        img /= 255
        print(img)

        # Save normalized image
        normalized_filename = os.path.join(normalized_folder_path, filename)
        cv2.imwrite(normalized_filename, img)

