from ultralytics import YOLO


class YoloTrain:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def train(self, data_path, is_absolute_path=False):
        if not is_absolute_path:
            import os
            data_path = os.path.abspath(data_path)



        self.model.train(
            data=data_path,    # 데이터 경로
            name='yolo_custom',  # 실험 이름
            pretrained=True,     # 사전학습 모델 사용 여부
            cfg='custom_config.yaml' # 하이퍼파라미터 파일
        )


if __name__ == "__main__":
    yolo_train = YoloTrain("yolov8n.yaml")
    yolo_train.train("Mechanical-tools-10000-3/data.yaml")