import torch
from ultralytics import YOLO
import cv2 as cv


model = YOLO("yolov8n.yaml")  # build a new model and move it to GPU if available

# Use the model
model.train(data="detect.yaml", epochs=27)  # train the model
