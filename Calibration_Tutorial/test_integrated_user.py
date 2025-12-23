import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG

import time
import numpy as np
import os
import json
import threading

import DR_init
from ultralytics import YOLO

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
# YOLO helper functions
# =========================
def build_model_path(project_folder: str, model_filename: str, base_dir: str | None = None) -> str:
    """
    - model_filename이 절대경로면 그대로 사용
    - 아니면 base_dir/project_folder/model_filename 형태로 구성
    """
    if os.path.isabs(model_filename):
        return model_filename

    if base_dir is None:
        base_dir = os.getcwd()

    output_dir = os.path.join(base_dir, project_folder)
    return os.path.join(output_dir, model_filename)


def load_yolo_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO 모델 파일이 없습니다: {model_path}")
    return YOLO(model_path)


def yolo_infer(model: YOLO, frame: np.ndarray, conf: float = 0.25):
    results = model.predict(source=frame, conf=conf, verbose=False)
    return results[0]


def draw_yolo_result(frame: np.ndarray, result) -> tuple[np.ndarray, list[dict]]:
    annotated = frame.copy()
    detections: list[dict] = []

    boxes = result.boxes
    names = result.names

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        name = str(names.get(cls_id, cls_id))
        label = f"{name} {conf*100:.1f}%"

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
                "name": name,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
                "center": (cx, cy),
            }
        )

    return annotated, detections


def yolo_infer_and_annotate(model: YOLO, frame: np.ndarray, conf: float = 0.25) -> tuple[np.ndarray, list[dict]]:
    result = yolo_infer(model, frame, conf=conf)
    return draw_yolo_result(frame, result)


# =========================
# Tool JSON helper functions
# =========================
TOOL_JSON_FILENAME = "tool_classes.json"
DEFAULT_TOOL_JSON = {
    "0": "apple",
    "1": "banana",
    "2": "pear",
    "3": "screwdriver",
    "4": "wrench",
}


def ensure_tool_json(json_dir: str) -> str:
    """
    json_dir/tool_classes.json 이 없으면 생성하고 경로를 반환합니다.
    """
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, TOOL_JSON_FILENAME)

    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_TOOL_JSON, f, ensure_ascii=False, indent=2)
        print(f"[INFO] created: {json_path}")

    return json_path


def load_tool_map(json_path: str) -> tuple[dict[int, str], dict[str, int]]:
    """
    {"0":"drill", ...} -> (id_to_name, name_to_id)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}

    for k, v in raw.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        name = str(v).strip().lower()
        id_to_name[idx] = name
        name_to_id[name] = idx

    return id_to_name, name_to_id


def normalize_user_target(user_input: str, id_to_name: dict[int, str]) -> str | None:
    """
    사용자 입력을 표준 타겟 문자열로 변환합니다.
    - 숫자(0~4): 그대로 매핑
    - 숫자(1~5): 사용자가 1부터 입력하는 습관을 고려해 (n-1)도 시도
    - 텍스트: 소문자 변환 후 그대로 사용
    """
    if user_input is None:
        return None

    s = user_input.strip().lower()
    if s == "":
        return None

    if s.isdigit():
        n = int(s)
        if n in id_to_name:
            return id_to_name[n]
        if (n - 1) in id_to_name:
            return id_to_name[n - 1]
        return None

    return s


# =========================
# Node
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

        # ---- YOLO 설정 ----
        self.project_folder = "Fruit"
        # ※ 기존 파일에서 사용하신 절대경로를 그대로 유지할 수 있습니다.
        #    상대경로면 project_folder와 결합됩니다.
        self.model_filename = "/home/rokey/Tutorial/roboflow_fruits/train_20251222_153819/weights/best.pt"
        self.yolo_conf = 0.25

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = build_model_path(self.project_folder, self.model_filename, base_dir=script_dir)

        self.get_logger().info(f"YOLO model path: {model_path}")
        self.yolo_model = load_yolo_model(model_path)

        self.last_detections: list[dict] = []

        # ---- Tool JSON 설정 ----
        tool_json_path = ensure_tool_json(script_dir)  # 스크립트 폴더에 생성
        self.id_to_tool, self.tool_to_id = load_tool_map(tool_json_path)

        # ---- 터미널 입력 쓰레드 ----
        self._req_lock = threading.Lock()
        self._requested_tool: str | None = None
        self.stop_event = threading.Event()
        self._picking = threading.Event()

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    # -------------------------
    # Input loop (terminal)
    # -------------------------
    def _print_help(self):
        print("\n====================================")
        print("사용법:")
        print(" - 이름으로 입력: drill / hammer / pliers / screwdriver / wrench")
        print(" - 숫자로 입력: 0~4 (또는 1~5도 자동 보정)")
        print(" - help: 목록 다시 보기")
        print(" - q   : 종료")
        print("------------------------------------")
        print("지원 목록:")
        for k in sorted(self.id_to_tool.keys()):
            print(f"  {k} : {self.id_to_tool[k]}")
        print("====================================\n")

    def _input_loop(self):
        self._print_help()

        while not self.stop_event.is_set():
            try:
                user_input = input("무엇을 집을까요? (name/number, help, q): ")
            except EOFError:
                self.stop_event.set()
                break

            s = (user_input or "").strip().lower()

            if s in ("q", "quit", "exit"):
                self.stop_event.set()
                break

            if s in ("help", "h", "?"):
                self._print_help()
                continue

            target = normalize_user_target(s, self.id_to_tool)
            if target is None or target not in self.tool_to_id:
                print(f"[NOT SUPPORTED] '{user_input}' 는 지원 목록에 없습니다.")
                continue

            with self._req_lock:
                self._requested_tool = target

            print(f"[REQUESTED] '{target}' 요청을 받았습니다. 화면에서 찾으면 집습니다.")

    def _pop_requested_tool(self) -> str | None:
        with self._req_lock:
            t = self._requested_tool
            self._requested_tool = None
        return t

    # -------------------------
    # Mouse click pick
    # -------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_frame = self._get_valid_depth_frame()

            print(f"img cordinate: ({x}, {y})")
            z = self.get_depth_value(x, y, depth_frame)
            if z is None or z <= 0:
                print("[DEPTH] 클릭 위치의 depth를 읽지 못했습니다.")
                return

            camera_center_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            print(f"camera cordinate: ({camera_center_pos})")

            robot_coordinate = self.transform_to_base(camera_center_pos)
            print(f"robot cordinate: ({robot_coordinate})")

            self.pick_and_drop(*robot_coordinate)
            print("=" * 100)

    # -------------------------
    # Camera / Transform helpers
    # -------------------------
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

    def transform_to_base(self, camera_coords):
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    # -------------------------
    # Depth utilities
    # -------------------------
    def _get_valid_depth_frame(self):
        depth_frame = self.img_node.get_depth_frame()
        tries = 0
        while (depth_frame is None or np.all(depth_frame == 0)) and tries < 20:
            self.get_logger().info("retry get depth img")
            rclpy.spin_once(self.img_node)
            time.sleep(0.05)
            depth_frame = self.img_node.get_depth_frame()
            tries += 1
        return depth_frame

    def get_depth_value(self, center_x, center_y, depth_frame):
        if depth_frame is None:
            return None
        height, width = depth_frame.shape
        if 0 <= center_x < width and 0 <= center_y < height:
            return float(depth_frame[int(center_y), int(center_x)])
        self.get_logger().warn(f"out of image range: {center_x}, {center_y}")
        return None

    def get_depth_median(self, x, y, depth_frame, window: int = 5):
        """
        중심 (x,y) 주변 window×window 영역의 '0이 아닌 값'의 중앙값을 반환합니다.
        (깊이 영상은 center 픽셀 하나가 0인 경우가 종종 있어서 안정화용)
        """
        if depth_frame is None:
            return None

        h, w = depth_frame.shape
        x = int(x)
        y = int(y)
        half = max(1, window // 2)

        xs = max(0, x - half)
        xe = min(w, x + half + 1)
        ys = max(0, y - half)
        ye = min(h, y + half + 1)

        roi = depth_frame[ys:ye, xs:xe].astype(np.float32)
        vals = roi[roi > 0]

        if vals.size == 0:
            return None

        return float(np.median(vals))

    # -------------------------
    # Robot actions
    # -------------------------
    def pick_and_drop(self, x, y, z):
        current_pos = get_current_posx()[0]
        pick_pos = posx([x, y, z+190, current_pos[3], current_pos[4], current_pos[5]])
        movel(pick_pos, vel=VELOCITY, acc=ACC)
        self.gripper.close_gripper()
        wait(1)

        movej(HOME_JOINT, vel=HOME_VEL, acc=HOME_ACC)
        self.gripper.open_gripper()
        wait(1)

    # -------------------------
    # Main image loop
    # -------------------------
    def open_img_node(self):
        rclpy.spin_once(self.img_node)
        img = self.img_node.get_color_frame()
        if img is None:
            return

        annotated, detections = yolo_infer_and_annotate(self.yolo_model, img, conf=self.yolo_conf)
        self.last_detections = detections

        # 마우스 클릭 좌표는 그대로 사용
        cv2.setMouseCallback("Webcam", self.mouse_callback, annotated)
        cv2.imshow("Webcam", annotated)

        # ---- 터미널 요청이 들어왔으면 자동 픽 시도 ----
        if self._picking.is_set():
            return

        target = self._pop_requested_tool()
        if not target:
            return

        # 현재 프레임 탐지 결과에서 target 찾기
        cand = [d for d in detections if str(d["name"]).strip().lower() == target]
        if not cand:
            print(f"[NOT FOUND] 현재 화면에서 '{target}' 을(를) 찾지 못했습니다.")
            return

        best = max(cand, key=lambda d: float(d["conf"]))
        cx, cy = best["center"]

        depth_frame = self._get_valid_depth_frame()
        z = self.get_depth_median(cx, cy, depth_frame, window=7)
        if z is None or z <= 0:
            print(f"[DEPTH] '{target}' 중심점 depth를 읽지 못했습니다. (다시 시도해 주세요)")
            return

        camera_pos = self.get_camera_pos(cx, cy, z, self.intrinsics)
        robot_xyz = self.transform_to_base(camera_pos)

        print(f"[PICK] '{target}' 을(를) 집으러 이동합니다. (pixel=({cx},{cy}), depth={z:.2f})")

        self._picking.set()
        try:
            self.pick_and_drop(*robot_xyz)
        finally:
            self._picking.clear()


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
        if test_node.stop_event.is_set():
            break

        test_node.open_img_node()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    test_node.stop_event.set()
    cv2.destroyAllWindows()
