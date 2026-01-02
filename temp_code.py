import sys
import time
import threading
import logging
import json

import rclpy
import DR_init
from rclpy.executors import MultiThreadedExecutor
from supabase import create_client, Client
from rclpy.callback_groups import ReentrantCallbackGroup

# [추가] 모션 제어용 서비스 임포트
from dsr_msgs2.srv import MovePause, MoveResume, MoveStop 

from ultralytics import YOLO
import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation
from .realsense import ImgNode
from .onrobot import RG
from statistics import mean
import time

# ==========================================
# 0. 로깅 설정 (System Logs)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ROBOT_MAIN")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ==========================================
# 1. 설정 및 상수 정의
# ==========================================
SUPABASE_URL = "https://mxslptjottitpveroyda.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im14c2xwdGpvdHRpdHB2ZXJveWRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY0ODk4MzEsImV4cCI6MjA4MjA2NTgzMX0.WQBiyDwRN1k02eetqjK0LLEFEvTA8M_16TGn0VM40tA"

# 로봇 식별 정보
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_DB_UUID = "3049c7a0-2ca2-417d-9cda-8b1e3bd597a9" 

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
VELOCITY, ACC = 100, 100

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# 테이블 이름
TABLE_ORDERS = "orders"
TABLE_ROBOT_STATE = "robot_status"

# Doosan Robot State 매핑
ROBOT_STATE_MAP = {
    0: "STATE_INITIALIZING", 1: "STATE_STANDBY", 2: "STATE_MOVING",
    3: "STATE_SAFE_OFF", 4: "STATE_TEACHING", 5: "STATE_SAFE_STOP",
    6: "STATE_EMERGENCY_STOP", 7: "STATE_HOMMING", 8: "STATE_RECOVERY",
    9: "STATE_SAFE_STOP2", 10: "STATE_SAFE_OFF2", 15: "STATE_NOT_READY",
}
ERROR_STATES = {3, 5, 6, 9, 10, 15}
RECOVERY_CONTROL_BY_STATE = {3: 3, 10: 3, 5: 2, 9: 2, 6: 1} # 자동 복구 매핑

# ==========================================
# 2. Database Manager Class (Supabase 통합 관리)
# ==========================================
class RobotDataManager:
    """
    Supabase DB와의 모든 통신을 전담하는 클래스
    - 데이터 조회, 상태 업데이트, 로깅을 담당
    """
    def __init__(self, url: str, key: str, robot_uuid: str):
        try:
            self.client = create_client(url, key)
            self.robot_uuid = robot_uuid
            logger.info("✅ Connected to Supabase")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            raise e

    def get_pending_order(self):
        """가장 오래된 'pending' 주문 1개 조회"""
        try:
            response = (
                self.client.table(TABLE_ORDERS)
                .select("id, yolo_names")
                .eq("robot_status", "pending")
                .order("created_at", desc=False) # 선입선출
                .limit(1)
                .execute()
            )

            if response.data:
                order = response.data[0]
                logger.info(f"📬 Found Order: {order['id']} | Items: {len(order.get('yolo_names', []))}")
                return order["id"], order.get("yolo_names", [])
            
            return None, None

        except Exception as e:
            logger.error(f"❌ Error fetching orders: {e}")
            return None, None

    def update_order_status(self, order_id: str, status: str):
        """주문 상태 업데이트 (성공 여부 로깅 포함)"""
        try:
            response = (
                self.client.table(TABLE_ORDERS)
                .update({"robot_status": status})
                .eq("id", order_id)
                .execute()
            )
            # data가 비어있지 않으면 업데이트 성공
            if response.data:
                logger.info(f"📝 Order({order_id}) Status Updated -> {status}")
            else:
                logger.warning(f"⚠️ Order({order_id}) update sent but no change detected.")

        except Exception as e:
            logger.error(f"❌ Failed to update order status: {e}")

    def update_robot_state(self, verbose=False, **fields):
        """
        로봇 모니터링 상태 DB 전송
        - 사용 가능 필드: is_working, current_task, error_code, error_message, desired_state 등
        - status, recovery_needed 제거됨
        """
        try:
            # 유효하지 않은 키 필터링 (안전장치)
            valid_keys = {"is_working", "current_task", "error_code", "error_message", "desired_state", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper_state"}
            filtered_fields = {k: v for k, v in fields.items() if k in valid_keys}
            
            if not filtered_fields:
                return

            response = (
                self.client.table(TABLE_ROBOT_STATE)
                .update(filtered_fields)
                .eq("id", self.robot_uuid)
                .execute()
            )
            
            if verbose and response.data:
                logger.info(f"📡 Robot State Updated to DB: {filtered_fields}")
            elif not response.data:
                # logger.warning(f"⚠️ Robot State update returned empty data. (UUID: {self.robot_uuid})")
                pass

        except Exception as e:
            logger.error(f"❌ Telemetry Error: {e}")

    def get_robot_desired_state(self):
        try:
            res = (
                self.client.table(TABLE_ROBOT_STATE)
                .select("desired_state")
                .eq("id", self.robot_uuid)
                .limit(1)
                .execute()
            )
            if not res.data:
                return ""
            row = res.data[0]
            desired = (row.get("desired_state") or "").strip().lower()
            return desired
        except Exception as e:
            logger.error(f"❌ get_robot_desired_state error: {e}")
            return ""

# ==========================================
# 3. Robot System & Motion Controller
# ==========================================
class MotionController:
    def __init__(self, node, robot_id: str):
        self.node = node
        self.cb_group = ReentrantCallbackGroup()

        self.pause_cli = node.create_client(MovePause, f"/{robot_id}/motion/move_pause", callback_group=self.cb_group)
        self.resume_cli = node.create_client(MoveResume, f"/{robot_id}/motion/move_resume", callback_group=self.cb_group)
        self.stop_cli = node.create_client(MoveStop, f"/{robot_id}/motion/move_stop", callback_group=self.cb_group)

        self._paused = False
        self._lock = threading.Lock()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def _set_paused(self, v: bool):
        with self._lock:
            self._paused = v

    def _call_wait(self, cli, req, timeout=2.0):
        fut = cli.call_async(req)
        start = time.time()
        while not fut.done() and (time.time() - start < timeout):
            time.sleep(0.01)
        return fut.result() if fut.done() else None

    def move_pause(self) -> bool:
        from dsr_msgs2.srv import MovePause
        res = self._call_wait(self.pause_cli, MovePause.Request())
        ok = bool(res is not None and getattr(res, "success", False))
        if ok:
            self._set_paused(True)
            self.node.get_logger().info("✅ MovePause 성공")
        else:
            self.node.get_logger().error("❌ MovePause 실패/타임아웃")
        return ok

    def move_resume(self) -> bool:
        from dsr_msgs2.srv import MoveResume
        res = self._call_wait(self.resume_cli, MoveResume.Request())
        ok = bool(res is not None and getattr(res, "success", False))
        if ok:
            self._set_paused(False)
            self.node.get_logger().info("✅ MoveResume 성공")
        else:
            self.node.get_logger().error("❌ MoveResume 실패/타임아웃")
        return ok

    def move_stop(self, stop_mode: int = 1) -> bool:
        from dsr_msgs2.srv import MoveStop
        req = MoveStop.Request()
        if hasattr(req, "stop_mode"):
            req.stop_mode = int(stop_mode)
        elif hasattr(req, "stop_type"):
            req.stop_type = int(stop_mode)

        res = self._call_wait(self.stop_cli, req)
        ok = bool(res is not None and getattr(res, "success", False))
        if ok:
            self.node.get_logger().warn(f"🚨 MoveStop 성공 (mode={stop_mode})")
        else:
            # MoveStop은 실패해도 치명적이지 않으므로(이미 멈춰있을 수 있음) warn 처리
            self.node.get_logger().warn("⚠️ MoveStop 실패/타임아웃 (이미 멈춰있거나 에러 상태일 수 있음)")
        return ok

class RobotSystemController:
    """
    외력 충돌 등으로 인한 Safe Stop 상태에서 자동 복구 담당
    """
    def __init__(self, node, robot_id: str):
        from DSR_ROBOT2 import GetRobotState, SetRobotControl
        self.node = node
        self.robot_id = robot_id
        self._GetRobotState = GetRobotState
        self._SetRobotControl = SetRobotControl
        self.cb_group = ReentrantCallbackGroup()
        
        self.state_cli = node.create_client(GetRobotState, f"/{robot_id}/system/get_robot_state", callback_group=self.cb_group)
        self.ctrl_cli = node.create_client(SetRobotControl, f"/{robot_id}/system/set_robot_control", callback_group=self.cb_group)

    def get_robot_state(self, timeout=1.0):
        if not self.state_cli.service_is_ready(): return -1
        req = self._GetRobotState.Request()
        fut = self.state_cli.call_async(req)
        
        start = time.time()
        while not fut.done():
            if time.time() - start > timeout: return -1
            time.sleep(0.01)
        if fut.result() is None: return -1
        return int(getattr(fut.result(), "robot_state", -1))

    def _standalone_recovery_step(self, target_control_mode: int) -> bool:
        temp_node = rclpy.create_node(f"recovery_{int(time.time()*1000)}")
        try:
            cli = temp_node.create_client(self._SetRobotControl, f"/{self.robot_id}/system/set_robot_control")
            if not cli.wait_for_service(timeout_sec=2.0): return False
            req = self._SetRobotControl.Request()
            req.robot_control = target_control_mode
            future = cli.call_async(req)
            rclpy.spin_until_future_complete(temp_node, future, timeout_sec=3.0)
            return True if (future.result() and future.result().success) else False
        finally:
            temp_node.destroy_node()

    def _standalone_get_state(self) -> int:
        temp_node = rclpy.create_node(f"state_chk_{int(time.time()*1000)}")
        try:
            cli = temp_node.create_client(self._GetRobotState, f"/{self.robot_id}/system/get_robot_state")
            if not cli.wait_for_service(timeout_sec=2.0): return -1
            future = cli.call_async(self._GetRobotState.Request())
            rclpy.spin_until_future_complete(temp_node, future, timeout_sec=2.0)
            return int(future.result().robot_state) if future.result() else -1
        finally:
            temp_node.destroy_node()

    def recover_if_possible(self, initial_state: int) -> bool:
        current_state = self._standalone_get_state()
        if current_state == -1: current_state = initial_state
        
        target_ctrl = RECOVERY_CONTROL_BY_STATE.get(current_state)
        if target_ctrl is None: return False

        logger.warning(f"⚡ Attempting Recovery: State {current_state} -> Cmd {target_ctrl}")
        
        if not self._standalone_recovery_step(target_ctrl):
            return False

        t_start = time.time()
        while time.time() - t_start < 10.0:
            time.sleep(1.0)
            s = self._standalone_get_state()
            
            if s not in ERROR_STATES and s > 0:
                logger.info(f"🎉 Recovery Complete! Current State: {s}")
                return True
            
            if s == 3: # Safe Off -> Servo On
                logger.info("⚡ Safe Off detected -> Sending Servo On")
                self._standalone_recovery_step(3)
                time.sleep(1.5)
                continue
            
            if s == 5 and (time.time() - t_start > 3.0):
                self._standalone_recovery_step(2) # Retry Reset

        return False

# ==========================================
# 4. 스레드 (상태 모니터링 & 명령 감시)
# ==========================================
def robot_state_monitor_thread(stop_event, fault_event, sysctl, db_manager: RobotDataManager):
    """
    로봇 상태 모니터링 스레드
    """
    last_state = -1

    while not stop_event.is_set():
        if fault_event.is_set():
            time.sleep(0.5) 
            continue

        state = sysctl.get_robot_state()
        if state != -1:
            # 상태 변경 여부 확인
            is_changed = (state != last_state)
            
            if is_changed:
                logger.info(f"🤖 Robot State Changed: {last_state} -> {state}")
                db_manager.update_robot_state(verbose=True, error_code=state)
                last_state = state
            else:
                db_manager.update_robot_state(verbose=False, error_code=state)

            # 에러 상태 감지
            if state in ERROR_STATES:
                if not fault_event.is_set():
                    logger.critical(f"🚨 FAULT DETECTED: State {state}")
                    # [수정] status, recovery_needed 제거 -> is_working, current_task로 대체
                    db_manager.update_robot_state(verbose=True, is_working=False, current_task="error", error_code=state)
                    fault_event.set()
        
        time.sleep(0.5)

# [수정] 명령 감시 스레드
# 파라미터 순서를 main()의 호출 순서와 일치시킴
def command_watcher_thread(
    run_gate: threading.Event,
    abort_event: threading.Event,
    home_event: threading.Event,
    stop_event: threading.Event,
    fault_event: threading.Event,      # [수정] main() 호출 순서에 맞게 5번째로 이동
    motion: MotionController,          # [수정] main() 호출 순서에 맞게 6번째로 이동
    db_manager: RobotDataManager,      # [추가] 7번째 파라미터로 추가
):
    """
    DB의 desired_state를 감시하여 pause/resume/stop/home 명령 처리
    
    [핵심 수정 사항]
    1. 파라미터 순서를 main()의 args 순서와 일치시킴
       - 기존: ..., motion, fault_event (잘못됨)
       - 수정: ..., fault_event, motion, db_manager
    2. 별도 supabase 연결 대신 db_manager 활용
    3. get_robot_desired_state() 대신 db_manager.get_robot_desired_state() 사용
    """
    last_cmd = None

    while not stop_event.is_set():
        try:
            # [수정] db_manager의 메서드 사용 (timestamp 불필요)
            cmd = db_manager.get_robot_desired_state()
            
            # 명령어가 바뀌었으면 새로운 명령으로 인식
            is_new = (cmd != last_cmd)

            if is_new and cmd:
                logger.info(f"[Watcher] New command: {cmd!r}")
                last_cmd = cmd

                # 1. Stop / Abort / Emergency
                if cmd in ("emergency_stop", "stop", "abort"):
                    logger.warning(f"🛑 {cmd.upper()} command received!")
                    if not fault_event.is_set():
                        motion.move_stop(stop_mode=1)
                    abort_event.set()
                    run_gate.set()  # Abort 처리를 위해 게이트 열어줌

                # 2. Home
                elif cmd == "home":
                    logger.info("🏠 HOME command received!")
                    if not fault_event.is_set():
                        motion.move_stop(stop_mode=1)
                    abort_event.set()
                    home_event.set()
                    run_gate.set()

                # 3. Pause
                elif cmd == "pause":
                    if not (abort_event.is_set() or fault_event.is_set() or home_event.is_set()):
                        logger.info("⏸️ PAUSE command received!")
                        # 로봇 모션 일시정지 서비스 호출
                        motion.move_pause()
                        # 게이트 닫기 → checkpoint()에서 대기하게 됨
                        run_gate.clear()

                # 4. Resume
                elif cmd == "resume":
                    if not (abort_event.is_set() or fault_event.is_set() or home_event.is_set()):
                        logger.info("▶️ RESUME command received!")
                        if motion.paused:
                            motion.move_resume()
                        if not run_gate.is_set():
                            run_gate.set()

                # 5. None/Working/기타 - 정상 상태 유지
                elif cmd in ("none", "working", ""):
                    # 일시정지 상태에서 DB가 초기화되면 자동 resume
                    if motion.paused and not run_gate.is_set():
                        logger.info("▶️ State cleared, auto-resuming...")
                        motion.move_resume()
                        run_gate.set()

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"❌ Watcher Error: {e}")
            time.sleep(1.0)

# ==========================================
# 5. yolo & Helper functions
# ==========================================
def load_yolo_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO 모델 파일이 없습니다: {model_path}")
    return YOLO(model_path)

def yolo_infer(model: YOLO, frame: np.ndarray, conf: float = 0.80):
    results = model.predict(source=frame, conf=conf, verbose=False)
    return results[0]

def draw_yolo_result(frame: np.ndarray, result):
    annotated = frame.copy()
    detections = []

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        detections.append({
            "cls_id": cls_id,
            "name": result.names[cls_id],
            "conf": conf,
            "center": (cx, cy),
            "x1y1x2y2":(x1,y1,x2,y2)
        })

    return annotated, detections

def select_target_detection(detections,target_class):
        candidates = [d for d in detections if d["name"] == target_class]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["conf"])

# ---------- Camera → Camera coord ----------
def get_camera_pos(cx, cy, z,img_node):
        intrinsics = img_node.get_camera_intrinsic()
        x = (cx - intrinsics["ppx"]) * z / intrinsics["fx"]
        y = (cy - intrinsics["ppy"]) * z / intrinsics["fy"]
        return np.array([x, y, z])

    # ---------- Camera → Robot base ----------
def transform_to_base(cam_coord,gripper2cam):
    from DSR_ROBOT2 import get_current_posx

    base2gripper = get_robot_pose_matrix(*get_current_posx()[0])
    base2cam = base2gripper @ gripper2cam

    cam_h = np.append(cam_coord, 1)
    base = base2cam @ cam_h
    return base[:3]

def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# ---------- Pick & Place Helper ----------
def select(x, y, z, gripper, check_func = None): 
    from DSR_ROBOT2 import movel, movesj, posx, posj, wait,get_current_posx,DR_MV_MOD_REL
    STOP_POINT1=posj([12.762, 6.61, 90.262, -0.04, 83.14, -79.958])#posx([400, 98.9, 150.801, 3.191, -179.976, -90.765])#
    STOP_POINT2=posj([-35.983, 8.636, 87.926, -0.051, 83.43, -125.702])#posx([350.865, -244.638, 150.453, 179.997, 179.978, 89.414])#
    STOP_POINT3=posj([-53.891, 29.944, 50.015, -0.164, 100.033, -143.681])#posx([340.637, -453.444, 206.405, 0.333, 179.319, -89.934])#
    
    if check_func: check_func()
    gripper.open_gripper()

    if check_func: check_func()
    
    current_pos = get_current_posx()[0]
    approach = posx([x, y-10, z+170, current_pos[3], current_pos[4], current_pos[5]])
    movel(approach,vel=VELOCITY,acc=ACC)
    if check_func: check_func()

    gripper.close_gripper()
    wait(1)
    if check_func: check_func()
    movel(posx([0, 0, 100, 0, 0, 0]),vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)

    if check_func: check_func()
    # movesx([STOP_POINT1,
    #        STOP_POINT2,
    #        STOP_POINT3], 
    #        vel=VELOCITY, acc=ACC)
    movesj([STOP_POINT1,
           STOP_POINT2,
           STOP_POINT3], 
           vel=VELOCITY, acc=ACC)    

def place(isExisted,gripper, check_func=None):
    from DSR_ROBOT2 import (movel, movesj, posx, posj, wait, DR_MV_MOD_REL, DR_FC_MOD_ABS,
                            task_compliance_ctrl, set_stiffnessx, set_desired_force, 
                            get_tool_force, release_force, release_compliance_ctrl)
    
    PLACE1_UP=posx([380.881, -596.913, -7.569, 85.326, 179.938, -4.787])
    PLACE2_UP=posx([367.111, -425.911, 3.807, 115.644, 179.991, 24.64])
    
    STOP_POINT1=posj([12.762, 6.61, 90.262, -0.04, 83.14, -79.958])#posx([400, 98.9, 150.801, 3.191, -179.976, -90.765])#
    STOP_POINT2=posj([-35.983, 8.636, 87.926, -0.051, 83.43, -125.702])#posx([350.865, -244.638, 150.453, 179.997, 179.978, 89.414])#
    if check_func: check_func()
    if isExisted:
        print("place2")
        movel(PLACE2_UP,vel=VELOCITY,acc=ACC)
    else:
        print("place1")
        movel(PLACE1_UP,vel=VELOCITY,acc=ACC)
    
    task_compliance_ctrl()
    compliance_on = True
    set_stiffnessx([3000, 3000, 5, 200, 200, 200], time=0)
    set_desired_force([0,0,-50,0,0,0],[0,0,1,0,0,0],time=0.0,mod=DR_FC_MOD_ABS)

    while True:
        force = get_tool_force()
        if check_func: check_func()
        if force[2] > 25:
            release_force(time=0)
            gripper.open_gripper() 
            wait(1)
            break

    release_compliance_ctrl()
    compliance_on = False
    if check_func: check_func()
    movel(posx([0, 0, 100, 0, 0, 0]),vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
    movesj([STOP_POINT2,
           STOP_POINT1], 
           vel=VELOCITY, acc=ACC)
    
import numpy as np

def stable_detect_with_bbox(img_node,yolo_model,target_class,frame_check=10,threshold=5):
    cx_list, cy_list = [], []
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    z_list = []

    start = time.time()
    for _ in range(frame_check):
        img = img_node.get_color_frame()
        depth = img_node.get_depth_frame()
        if img is None or depth is None:
            continue

        result = yolo_infer(yolo_model, img)
        _, detections = draw_yolo_result(img, result)

        target = select_target_detection(detections, target_class)
        if target is None:
            continue

        cx, cy = map(int, target["center"])
        h, w = depth.shape
        x_1 = max(cx - 1, 0)
        x_2 = min(cx + 2, w)
        y_1 = max(cy - 1, 0)
        y_2 = min(cy + 2, h)

        cx_list.append(cx)
        cy_list.append(cy)
        x1,y1,x2,y2=target["x1y1x2y2"]
        x1_list.append(x1)
        y1_list.append(y1)
        x2_list.append(x2)
        y2_list.append(y2)
        
        roi = depth[y_1:y_2, x_1:x_2].astype(np.int32)
        valid_depths = roi[(roi > 0) & (roi < 3000)]

        if valid_depths.size > 0:
            z_list.append(np.median(valid_depths))

    if len(z_list) > 0:
            last_z = int(np.median(z_list))  # 프레임 간도 median
            print("stable depth:", last_z)
    else:
            print("depth measurement failed")
    end = time.time()
    print(f"start :{start} - end :{end} = {end - start:.5f} sec")
    if len(cx_list) < threshold:
        print(len(cx_list))
        return None
    
    z = depth[cy, cx]
    if abs(last_z-int(z))>10:
            logger.info(f"z={z}, last_z={last_z}, diff={abs(last_z-z)}")
    
    logger.info(f"🔍 Object still detected ({len(cx_list)}/{threshold})")
    stable_target = {
        "center": (
            int(np.median(cx_list)),
            int(np.median(cy_list))
        ),
        "x1y1x2y2": (
            int(np.median(x1_list)),
            int(np.median(y1_list)),
            int(np.median(x2_list)),
            int(np.median(y2_list))
        ),
        "depth":(
            last_z
        )
    }

    return stable_target


def go_home_motion():
    from DSR_ROBOT2 import movej, posj
    # 기본 홈 위치 정의
    HOME_JOINT = [0, 0, 90, 0, 90, 0] 
    logger.info(f"🏠 Going HOME {HOME_JOINT}")
    movej(posj(HOME_JOINT), vel=60, acc=60)

# ==========================================
# 6. 작업함수
# ==========================================
class StopRequested(Exception): pass

def perform_task(
    yolo_names: list, 
    fault_event: threading.Event, 
    abort_event: threading.Event,
    run_gate: threading.Event, 
    img_node,
    yolo_model,      
    gripper2cam,     
    gripper  
) -> tuple[list, list]:
    
    print("*"*50)
    from DSR_ROBOT2 import movej, posj, wait, DR_MV_MOD_REL
    isLong = False
    completed_items = []
    failed_items = []
    print("*"*50)
    
    # [수정] 체크포인트에서 일시정지(run_gate) 확인
    def checkpoint():
        if fault_event.is_set(): raise StopRequested("External Force/Fault Detected")
        if abort_event.is_set(): raise StopRequested("Abort Requested")
        # logger.info("Checking run_gate...")
        # Pause 상태면 여기서 대기
        if not run_gate.is_set():
            logger.info("⏸️ Task Paused. Waiting for Resume...")
            run_gate.wait()
            # 깨어났는데 Abort일 수도 있으니 다시 체크
            if abort_event.is_set(): raise StopRequested("Abort Requested during Pause")
            logger.info("▶️ Task Resumed.")

    logger.info(f"🚀 Start Processing: {yolo_names}")
    TABLE_UP=posj([60, 19.319, 49.142, 0.416, 107.438, 63.165]) #posx([290.0, 409.975, 309.514, 60.864, 175.867, 70.147])
    checkpoint()
    
    for item in yolo_names:
        movej(TABLE_UP, vel=VELOCITY, acc=ACC)
        wait(1)
        checkpoint() # 이동 후 체크

        logger.info(f"   -> Acting on item: {item}")

        start = time.time()
        img, depth = None, None
        img = img_node.get_color_frame()
        depth = img_node.get_depth_frame()
        if img is None or depth is None:
            logger.error(f"Camera frame failed for : {item}")
            failed_items.append(item)
            continue

        result = yolo_infer(yolo_model, img)
        annotated, detections= draw_yolo_result(img, result)
        cv2.imshow("YOLO", annotated)

        target_class=item
        print("-"*50)
        print(select_target_detection(detections,target_class))
        target = stable_detect_with_bbox(img_node,yolo_model,item)
        print(target)
        if target is None:
            logger.info("❌ target not stably detected")
            failed_items.append(item)
            continue
        cx,cy = target["center"]
        z = target["depth"]
        
        cam_pos = get_camera_pos(cx, cy, z,img_node)
        robot_pos = transform_to_base(cam_pos,gripper2cam)
        
        x1,y1,x2,y2 = target["x1y1x2y2"]
        if abs(x1-x2)>abs(y1-y2):
            isLong = True

        checkpoint() # Pick 전 체크

        if isLong:
            movej(posj([0, 0, 0, 0, 0, -90]),vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            wait(0.5)
            print("turn")

        print("grip start")
        select(*robot_pos,gripper, check_func=checkpoint)
        print("pickup end")
        checkpoint() # Place 전 체크

        FRAME_CHECK = 10
        DETECT_THRESHOLD = 5

        detected_count = 0
        valid_frame_count = 0

        for _ in range(FRAME_CHECK):
            img2 = img_node.get_color_frame()

            if img2 is None:
                logger.warning("⚠️ Camera frame unavailable during place check")
                continue

            result2 = yolo_infer(yolo_model, img2)
            num_objects = len(result2.boxes)

            valid_frame_count += 1

            if num_objects > 1:
                detected_count += 1

        isExisted = False

        if valid_frame_count == 0:
            logger.warning("❌ No valid frames for object check")
            isExisted = False
        elif detected_count >= DETECT_THRESHOLD:
            isExisted = True
            logger.info(f"🔍 Object still detected ({detected_count}/{valid_frame_count})")
        else:
            logger.info(f"✅ Object not detected ({detected_count}/{valid_frame_count}) → assumed picked")

        place(isExisted,gripper, check_func=checkpoint)
        print("place end")
        completed_items.append(item)
        checkpoint() # 루프 마지막 체크

    movej(TABLE_UP, vel=60, acc=60)
    logger.info("✅ Task Finished.")
    return completed_items, failed_items
    
# ==========================================
# 7. 메인 함수
# ==========================================
def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("order_processor", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    # 1. Executor 실행
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()
    
    yolo_model = load_yolo_model(
            "/home/archer/ros2_ws/src/pre_project/pre_project/best.pt"
        )
    gripper2cam = np.load(
            "/home/archer/ros2_ws/src/pre_project/pre_project/T_gripper2camera.npy"
        )
    gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
    img_node = ImgNode()
    executor.add_node(img_node)

    logger.info("Waiting for camera...")
    wait_start = time.time()

    while time.time() - wait_start < 10.0:
        img = img_node.get_color_frame()
        depth = img_node.get_depth_frame()
        if img is not None and depth is not None:
            logger.info("Camera Ready!")
            break
        time.sleep(0.5)
    else:
        logger.error("Camera initialization timeout!")
        return

    try:
        db_manager = RobotDataManager(SUPABASE_URL, SUPABASE_KEY, ROBOT_DB_UUID)
    except Exception:
        logger.critical("DB Connection Failed. Exiting.")
        return

    try:
        motion = MotionController(node, ROBOT_ID)
        logger.info("✅ Motion Controller Ready")
    except Exception as e:
        logger.error(f"❌ Motion Init failed: {e}")
        return

    try:
        sysctl = RobotSystemController(node, ROBOT_ID)
        logger.info("🤖 System Controller Ready")
    except Exception as e:
        logger.error(f"❌ Init failed: {e}")
        return

    # 이벤트 정의
    stop_event = threading.Event()
    fault_event = threading.Event()
    abort_event = threading.Event()
    home_event = threading.Event()
    run_gate = threading.Event() 
    run_gate.set() 

    # 상태 모니터 스레드
    monitor = threading.Thread(
        target=robot_state_monitor_thread,
        args=(stop_event, fault_event, sysctl, db_manager),
        daemon=True
    )
    monitor.start()

    # 명령 감시 스레드
    watcher = threading.Thread(
        target=command_watcher_thread,
        args=(run_gate, abort_event, home_event, stop_event, fault_event, motion, db_manager),
        daemon=True
    )
    watcher.start()

    logger.info("👀 Waiting for orders...")

    try:
        while rclpy.ok():
            # A. 에러 복구
            if fault_event.is_set():
                logger.warning("🚑 Handling Fault...")
                # [수정] status, recovery_needed 제거 -> is_working, current_task 업데이트
                db_manager.update_robot_state(verbose=True, is_working=True, current_task="recovering")
                
                success = sysctl.recover_if_possible(initial_state=5)
                
                if success:
                    logger.info("✅ Resuming normal operation.")
                    fault_event.clear()
                    # [수정] is_working=False, current_task="idle"
                    db_manager.update_robot_state(verbose=True, is_working=False, current_task="idle")
                else:
                    logger.error("⛔ Manual Recovery Required.")
                    time.sleep(2.0)
                continue
            
            # [추가] Home 요청 처리
            if home_event.is_set():
                logger.info("🏠 Home Event Triggered - Moving to Home")
                try:
                    # 일시정지 상태라면 해제하고 이동
                    if motion.paused: 
                        motion.move_resume()
                    go_home_motion()
                    # [수정] status -> current_task
                    db_manager.update_robot_state(desired_state="None", is_working=False, current_task="idle")
                except Exception as e:
                    logger.error(f"❌ Home Move Failed: {e}")
                    # [수정] status -> current_task, error_code
                    db_manager.update_robot_state(is_working=False, current_task="error")
                finally:
                    home_event.clear()
                    abort_event.clear()
                    run_gate.set()

            # B. 주문 처리
            if not (abort_event.is_set() or home_event.is_set()):
                order_id, yolo_names = db_manager.get_pending_order()
                
                if order_id:
                    db_manager.update_order_status(order_id, "picking")
                    # [수정] status="working" -> is_working=True, current_task="picking"
                    db_manager.update_robot_state(is_working=True, current_task="picking")
                    
                    try:
                        completed, failed = perform_task(
                            yolo_names, fault_event, abort_event, run_gate,
                            img_node, yolo_model, gripper2cam, gripper
                        )
                        if len(failed) == 0:
                            db_manager.update_order_status(order_id, "completed")
                        else:
                            db_manager.update_order_status(order_id, "failed")
                        
                        # [수정] status="idle" -> is_working=False
                        db_manager.update_robot_state(is_working=False, current_task="idle")

                    except StopRequested as e:
                        logger.warning(f"🛑 Task Interrupted: {e}")
                        db_manager.update_order_status(order_id, "failed")
                        # [수정] status="idle"
                        db_manager.update_robot_state(is_working=False, current_task="idle")
                        abort_event.clear() 
                    
                    except Exception as e:
                        logger.error(f"❌ Task Error: {e}")
                        db_manager.update_order_status(order_id, "failed")
                        # [수정] status="error"
                        db_manager.update_robot_state(is_working=False, current_task="error")
            
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Closing System...")
    finally:
        stop_event.set()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
