import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG

import time
import numpy as np
import os

import DR_init
from ultralytics import YOLO  # :white_check_mark: 추가

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

HOME_JOINT = [0, 0, 90, 0, 90, 0]
HOME_VEL = 60
HOME_ACC = 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


# =========================
# :white_check_mark: YOLO 함수화 (로드/추론/시각화)
# =========================
def build_model_path(project_folder: str, model_filename: str, base_dir: str | None = None) -> str:
    """
    project_folder/best.pt 형태 경로를 만들어 반환합니다.
    base_dir를 안 주면 현재 작업 디렉토리(os.getcwd()) 기준입니다.
    """
    if base_dir is None:
        base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, project_folder)
    return os.path.join(output_dir, model_filename)


def load_yolo_model(model_path: str) -> YOLO:
    """YOLO 모델 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO 모델 파일이 없습니다: {model_path}")
    return YOLO(model_path)


def yolo_infer(model: YOLO, frame: np.ndarray, conf: float = 0.25):
    """
    프레임 1장 추론 결과(result 1개)를 반환합니다.
    """
    results = model.predict(source=frame, conf=conf, verbose=False)
    return results[0]  # 단일 프레임이므로 0번


def draw_yolo_result(frame: np.ndarray, result) -> tuple[np.ndarray, list[dict]]:
    """
    result(boxes 포함)를 받아 frame에 박스를 그리고,
    detections 리스트도 함께 반환합니다.

    detections 예:
      [{"cls_id":0,"name":"apple","conf":0.87,"xyxy":(x1,y1,x2,y2),"center":(cx,cy)}, ...]
    """
    annotated = frame.copy()
    detections: list[dict] = []

    boxes = result.boxes
    names = result.names  # 클래스 id -> 이름

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        label = f"{names[cls_id]} {conf*100:.1f}%"

        # 박스/텍스트 표시
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        detections.append(
            {
                "cls_id": cls_id,
                "name": names[cls_id],
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
                "center": (cx, cy),
            }
        )

    return annotated, detections


def yolo_infer_and_annotate(model: YOLO, frame: np.ndarray, conf: float = 0.25) -> tuple[np.ndarray, list[dict]]:
    """한 방에: 추론 + 그리기"""
    result = yolo_infer(model, frame, conf=conf)
    return draw_yolo_result(frame, result)


# =========================
# 마우스 콜백 + 로봇 제어 노드
# =========================
class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self.img_node = ImgNode()
        rclpy.spin_once(self.img_node)
        time.sleep(1)

        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.gripper2cam = np.load("T_gripper2camera.npy")
        self.JReady = posj([0, 0, 90, 0, 90, -90])
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        # :white_check_mark: YOLO 설정/로드 (원하시는 폴더/파일명 그대로)
        self.project_folder = "Fruit"
        self.model_filename = "/home/rokey/Tutorial/roboflow_fruits/train_20251222_153819/weights/best.pt"
        self.yolo_conf = 0.25

        # 보통 스크립트 위치 기준이 안전합니다(실행 폴더가 바뀌어도 경로가 안 깨짐)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = build_model_path(self.project_folder, self.model_filename, base_dir=script_dir)

        self.get_logger().info(f"YOLO model path: {model_path}")
        self.yolo_model = load_yolo_model(model_path)

        # (선택) 마지막 탐지 결과 저장해두면, 나중에 자동 픽 같은 기능 추가하기 쉬움
        self.last_detections: list[dict] = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_frame = self.img_node.get_depth_frame()
            while depth_frame is None or np.all(depth_frame == 0):
                self.get_logger().info("retry get depth img")
                rclpy.spin_once(self.img_node)
                depth_frame = self.img_node.get_depth_frame()

            print(f"img cordinate: ({x}, {y})")
            z = self.get_depth_value(x, y, depth_frame)
            camera_center_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            print(f"camera cordinate: ({camera_center_pos})")

            robot_coordinate = self.transform_to_base(camera_center_pos)
            print(f"robot cordinate: ({robot_coordinate})")

            self.pick_and_drop(*robot_coordinate)
            print("=" * 100)

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z
        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def pick_and_drop(self, x, y, z):
        current_pos = get_current_posx()[0]
        pick_pos = posx([x, y, z+190, current_pos[3], current_pos[4], current_pos[5]])
        movel(pick_pos, vel=VELOCITY, acc=ACC)
        self.gripper.close_gripper()# movel(...)
        wait(1)

        # movej(...)  # move to initial position
        movej(HOME_JOINT, vel=HOME_VEL, acc=HOME_ACC)
        self.gripper.open_gripper()
        wait(1)

    def transform_to_base(self, camera_coords):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def open_img_node(self):
        rclpy.spin_once(self.img_node)
        img = self.img_node.get_color_frame()
        if img is None:
            return

        # :white_check_mark: YOLO 박스 표시된 화면으로 출력
        annotated, detections = yolo_infer_and_annotate(self.yolo_model, img, conf=self.yolo_conf)
        self.last_detections = detections[0]
        print("="*50)
        print(self.last_detections["center"])
        last_center = self.last_detections["center"]

        depth_frame = self.img_node.get_depth_frame()
        while depth_frame is None or np.all(depth_frame == 0):
            self.get_logger().info("retry get depth img")
            rclpy.spin_once(self.img_node)
            depth_frame = self.img_node.get_depth_frame()

        z = self.get_depth_value(last_center[0], last_center[1], depth_frame)
        camera_center_pos = self.get_camera_pos(last_center[0], last_center[1], z, self.intrinsics)

        robot_coordinate = self.transform_to_base(camera_center_pos)
        self.pick_and_drop(*robot_coordinate)

        # 마우스 클릭 좌표는 그대로 사용 (클릭하면 그 픽셀의 depth로 pick 수행)
        cv2.setMouseCallback("Webcam", self.mouse_callback, annotated)
        cv2.imshow("Webcam", annotated)

    def get_depth_value(self, center_x, center_y, depth_frame):
        height, width = depth_frame.shape
        if 0 <= center_x < width and 0 <= center_y < height:
            depth_value = depth_frame[center_y, center_x]
            return depth_value
        self.get_logger().warn(f"out of image range: {center_x}, {center_y}")
        return None


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import (
            get_current_posx,
            movej,
            movel,
            wait,
        )
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        exit(True)

    cv2.namedWindow("Webcam")

    test_node = TestNode()

    while True:
        test_node.open_img_node()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()