#Atul R, Basic Object Detection using YOLOv5, OpenCV and a given model

import torch
import cv2 as cv

# Load the model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.load_state_dict(torch.load("v3_20fps.pt", map_location=torch.device('cpu')))
model.eval()

# Set up video capture
video_path = "2744.mp4"
cap = cv.VideoCapture(video_path)
frame_rate = int(cap.get(cv.CAP_PROP_FPS))

# Read the video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    # Convert the frame 
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Get the model predictions from the frame
    results = model(rgb_frame)

    # Process the results and get bounding boxes
    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, class_id = result.tolist()
        
       # Filter detections by confidence 
        if conf > 0.2:  
            # Draw bounding box
            cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Display the detection class and confidence
            label = f'Class {int(class_id)}: {conf:.2f}'
            cv.putText(frame, label, (int(xmin), int(ymin) - 10),
                       cv.FONT_ITALIC, 0.5, (0, 255, 0), 2)

    # Display the video with the detection
    cv.imshow('Vision Application Challenge', frame)

    # Exit on pressing 'q'
    if cv.waitKey(frame_rate) & 0xFF == ord('q'):
        break

# Release the and close once finished
cap.release()
cv.destroyAllWindows()