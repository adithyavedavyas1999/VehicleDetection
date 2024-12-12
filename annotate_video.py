from ultralytics import YOLO
import cv2

# Input video path
video_path = "/content/drive/MyDrive/archive (1)/TestVideo/TrafficPolice.mp4"

# Open the video file for reading
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Define output video settings
output_path = "/content/drive/MyDrive/annotated_video_0.4thres.mp4"  # Path to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        print("End of video or error reading frame.")
        break

    # Perform object detection on the current frame and Set confidence threshold for filtering predictions
    results = model.predict(frame, conf=0.4)

    # Annotate the frame with detected objects
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            confidence = box.conf[0]  # Confidence score of the prediction
            class_id = int(box.cls[0])  # Class ID of the detected object

            # Prepare label with class name and confidence score
            label = f"{model.names[class_id]} {confidence:.2f}"
            color = (0, 255, 0)  # Bounding box color 

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw rectangle
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Add label

    # Write the annotated frame to the output video file
    out.write(frame)

cap.release()  # Close the input video
out.release()  # Close the output video writer
cv2.destroyAllWindows()  

print(f"Annotated video saved to {output_path}")