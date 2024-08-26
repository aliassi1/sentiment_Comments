import os
from ultralytics import YOLO
import cv2

video_path = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\vid1.mp4"
output_directory = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\comments_images"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5
frame_number = 0

while ret:
    frame_number += 1
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
                # Extract the region of interest (ROI) where the object is detected
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Save the ROI as an image
            cv2.imwrite(os.path.join(output_directory, f"comment_frame_{frame_number}.jpg"), roi)

            # Draw bounding box and label on the original frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
