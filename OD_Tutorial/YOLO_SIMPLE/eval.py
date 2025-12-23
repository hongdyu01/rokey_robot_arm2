from ultralytics import YOLO

# 사전학습된 YOLO 모델 로드 (예: YOLOv8n)
model = YOLO("yolov8n.pt")

# 이미지에 대해 inference 실행
results = model(["sample2.jpg", "sample.jpg"], imgsz=640)

# 결과 시각화

for result in results:
    result.show()

# 추가적으로 결과 정보 출력 (옵션)
print(results)