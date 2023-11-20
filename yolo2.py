import os
import cv2
import torch

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Folder containing the images
input_folder = 'images'
output_folder = 'detect'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all files in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')): # Add other file types if needed
        # Read the image
        path = os.path.join(input_folder, filename)
        image = cv2.imread(path)

        # Perform inference
        results = model(image)

        # Render the results on the image
        frame_with_results = results.render()[0]

        # Save the image with detections
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, frame_with_results)

print("Processing complete.")
