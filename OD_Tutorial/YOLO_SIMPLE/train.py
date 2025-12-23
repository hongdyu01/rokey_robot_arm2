from ultralytics import YOLO
import os

model = YOLO("yolov8n-det.pt")
model.train(data="datasets_seg/data.yaml", epochs=300, save_period=10, name="yolov8_tool_seg_0123")
