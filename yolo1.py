import cv2
import torch

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Get the original frame rate of the webcam
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object (double the FPS for 2x speed)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, original_fps * 2, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Render the results on the image
    frame_with_results = results.render()[0]

    # Write the frame into the file 'output.mp4'
    out.write(frame_with_results)

    # Display the image
    cv2.imshow('YOLOv5 Webcam Object Detection', frame_with_results)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
