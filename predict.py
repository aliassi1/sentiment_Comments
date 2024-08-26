import os
from ultralytics import YOLO
import cv2
import easyocr as ez

video_path = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\vid1.mp4"
output_directory = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\comments_images"
video_path_out = '{}_out.mp4'.format(video_path)


reader = ez.Reader(['en'], gpu=True)
comments = set()

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))



model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')




# Load a model
model = YOLO(r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\runs\detect\train4\weights\last.pt")  # load a custom model

threshold = 0.5
frame_number = 0
frame_rate = 30  # Video frame rate (fps)
target_frame_rate = 30    # Target frame rate (frames per second)
target_interval = int(frame_rate / target_frame_rate)

while ret:
    frame_number += 1

    # If the current frame number is a multiple of the target interval, process the frame
    if frame_number % target_interval == 0:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                # Extract the region of interest (ROI) where the object is detected
                roi = frame[int(y1):int(y2), int(x1):int(x2)]

                # Read text from ROI
                text = reader.readtext(roi)
                for t in text:
                    if t[2] > 0.5:
                        # Normalize the text by converting to lowercase and removing leading/trailing whitespaces
                         normalized_text = t[1].replace(" ", "")

                        # Check if the normalized text already exists in the set of comments
                         if normalized_text not in comments:
                            # Add the normalized text to the set of comments
                            comments.add(t[1])

                            # Save the ROI as an image
                            image_name = f"comment_{len(comments)}.jpg"
                            image_path = os.path.join(output_directory, image_name)
                            
                            video_pathhh = os.path.join(output_directory, image_name)

                            cv2.imwrite(image_path, roi)
                            cv2.imwrite(video_pathhh, roi)


                # Draw bounding box and label on the original frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)   
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
print("Total unique comments captured:", len(comments))
print(comments)