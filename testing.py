import os
from ultralytics import YOLO
import cv2
import easyocr as ez
import os
import cv2
import easyocr as ez
import joblib
import pandas as pd


reg = joblib.load(r"C:\Users\AliOs\Downloads\comments_model.joblib")

image_path = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\data\images\train\IMG_0224.PNG"
output_directory = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\comments_images"

reader = ez.Reader(['en'], gpu=True)
comments = set()

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load image
frame = cv2.imread(image_path)

model_path = r"C:\Users\AliOs\OneDrive\Desktop\COMP VISION\runs\detect\train4\weights\last.pt"

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Process the frame
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
                    cv2.imwrite(image_path, roi)

                    # Draw bounding box and label on the original frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the annotated image
annotated_image_path = os.path.join(output_directory, "annotated_image.jpg")
cv2.imwrite(annotated_image_path, frame)

print("Total unique comments captured:", len(comments))
print("Comments:", comments)

comments=list(comments)
df_topredict=pd.DataFrame({'comments':comments})
vectorizerrr=joblib.load(r"C:\Users\AliOs\Downloads\tfidf.joblib")
print(reg.predict(vectorizerrr.transform(df_topredict['comments'])))


