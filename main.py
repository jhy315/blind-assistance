# -*- coding: utf-8 -*-
"""
盲人出行辅助提醒系统 v2.0
功能：基于YOLOv8实时检测障碍物和路口，语音播报提醒
"""

from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading
import queue
import sys
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ==================== 中文字体配置 ====================
def get_chinese_font(size=20):
    """获取中文字体"""
    # Windows 常见中文字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/simkai.ttf",      # 楷体
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    
    # 如果找不到字体，使用默认字体
    return ImageFont.load_default()


# ==================== 配置参数 ====================
class Config:
    """系统配置参数"""
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CONF_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.5
    BARRIER_GAP = 3
    CROSS_GAP = 5
    PERSON_GAP = 4
    NEAR_THRESHOLD = 0.15
    MID_THRESHOLD = 0.05
    LEFT_BOUND = 0.35
    RIGHT_BOUND = 0.65


# ==================== 障碍物分类 ====================
class ObjectCategories:
    """目标分类定义"""
    BARRIERS = {
        "car", "truck", "bus", "bicycle", "motorcycle",
        "bench", "chair", "couch", "potted plant", "fire hydrant",
        "stop sign", "parking meter", "suitcase", "backpack",
        "handbag", "sports ball", "skateboard", "surfboard",
        "snowboard", "kite", "baseball bat", "baseball glove",
        "skis", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush", "book", "clock", "umbrella",
        "cell phone", "laptop", "mouse", "remote", "keyboard",
        "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
        "blender", "dog", "cat", "bird", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe"
    }
    CROSSING = {"traffic light", "stop sign"}
    PEDESTRIANS = {"person"}


# ==================== 中文绘制工具 ====================
class ChineseDrawer:
    """用于在OpenCV图像上绘制中文"""
    
    def __init__(self):
        self.font_small = get_chinese_font(16)
        self.font_medium = get_chinese_font(20)
        self.font_large = get_chinese_font(24)
    
    def put_chinese_text(self, img, text, position, font_size=20, color=(255, 255, 255)):
        """
        在OpenCV图像上绘制中文
        img: OpenCV图像 (numpy array)
        text: 中文文本
        position: (x, y) 位置
        font_size: 字体大小
        color: BGR颜色
        """
        # 转换颜色从BGR到RGB
        color_rgb = (color[2], color[1], color[0])
        
        # OpenCV转PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 选择字体
        if font_size <= 16:
            font = self.font_small
        elif font_size <= 20:
            font = self.font_medium
        else:
            font = self.font_large
        
        # 绘制文字
        draw.text(position, text, font=font, fill=color_rgb)
        
        # PIL转回OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==================== 语音播报线程 ====================
class VoiceAnnouncer(threading.Thread):
    """独立线程处理语音播报"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.speech_queue = queue.Queue()
        self.running = True
        self.last_messages = {}
        self.min_repeat_gap = 2
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 170)
            self.engine.setProperty('volume', 0.9)
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            print("[语音系统] 初始化成功")
        except Exception as e:
            print(f"[语音系统] 初始化失败: {e}")
            self.engine = None
    
    def speak(self, message, priority=0):
        if not self.engine:
            return
        now = time.time()
        if message in self.last_messages:
            if now - self.last_messages[message] < self.min_repeat_gap:
                return
        self.last_messages[message] = now
        self.speech_queue.put((priority, message, now))
    
    def run(self):
        while self.running:
            try:
                priority, message, _ = self.speech_queue.get(timeout=1)
                if self.engine:
                    self.engine.say(message)
                    self.engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[语音错误] {e}")
    
    def stop(self):
        self.running = False
        if self.engine:
            self.engine.stop()
        self.join(timeout=2)


# ==================== 距离估算器 ====================
class DistanceEstimator:
    """基于目标框大小估算距离"""
    
    @staticmethod
    def estimate(frame_area, box_area):
        ratio = box_area / frame_area
        if ratio > Config.NEAR_THRESHOLD:
            return "近距离"
        elif ratio > Config.MID_THRESHOLD:
            return "中距离"
        else:
            return "远距离"
    
    @staticmethod
    def get_direction(frame_width, center_x):
        ratio = center_x / frame_width
        if ratio < Config.LEFT_BOUND:
            return "左侧"
        elif ratio > Config.RIGHT_BOUND:
            return "右侧"
        else:
            return "正前方"


# ==================== 主程序 ====================
class BlindAssistanceSystem:
    """盲人出行辅助提醒系统主类"""
    
    def __init__(self):
        self.model = None
        self.cap = None
        self.voice = None
        self.distance_estimator = DistanceEstimator()
        self.chinese_drawer = ChineseDrawer()  # 中文绘制工具
        
        self.last_barrier_time = 0
        self.last_cross_time = 0
        self.last_person_time = 0
        self.detection_history = deque(maxlen=5)
        self.fps_history = deque(maxlen=30)
        self.prev_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
    
    def initialize(self):
        """初始化所有组件"""
        print("=" * 50)
        print("  盲人出行辅助提醒系统 v2.0")
        print("=" * 50)
        
        try:
            print("[模型加载] 正在加载 YOLOv8n...")
            self.model = YOLO("yolov8n.pt")
            print("[模型加载] 成功")
        except Exception as e:
            print(f"[模型加载] 失败: {e}")
            return False
        
        try:
            print("[摄像头] 正在初始化...")
            self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            ret, _ = self.cap.read()
            if not ret:
                raise Exception("无法读取摄像头画面")
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[摄像头] 成功: {actual_width}x{actual_height}")
        except Exception as e:
            print(f"[摄像头] 失败: {e}")
            return False
        
        try:
            self.voice = VoiceAnnouncer()
            self.voice.start()
            time.sleep(0.5)
        except Exception as e:
            print(f"[语音系统] 启动失败: {e}")
            return False
        
        print("-" * 50)
        print("系统已启动，按 Q 键退出")
        print("-" * 50)
        return True
    
    def process_frame(self, frame):
        """处理单帧画面"""
        h, w = frame.shape[:2]
        frame_area = h * w
        
        results = self.model(frame, conf=Config.CONF_THRESHOLD, iou=Config.IOU_THRESHOLD, verbose=False)
        
        current_detections = {
            "barriers": [],
            "crossings": [],
            "pedestrians": []
        }
        
        annotated_frame = frame.copy()
        
        for res in results:
            boxes = res.boxes
            names = res.names
            
            for box in boxes:
                cls_id = int(box.cls[0])
                obj_name = names[cls_id]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                distance = self.distance_estimator.estimate(frame_area, box_area)
                direction = self.distance_estimator.get_direction(w, center_x)
                
                # 中文类别名称映射
                chinese_names = {
                    "person": "行人", "car": "汽车", "truck": "卡车", "bus": "公交车",
                    "bicycle": "自行车", "motorcycle": "摩托车", "bench": "长椅",
                    "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
                    "fire hydrant": "消防栓", "stop sign": "停车标志",
                    "parking meter": "停车计时器", "traffic light": "红绿灯"
                }
                display_name = chinese_names.get(obj_name, obj_name)
                
                if obj_name in ObjectCategories.BARRIERS:
                    current_detections["barriers"].append((distance, direction, obj_name, conf))
                    color = (0, 0, 255)
                elif obj_name in ObjectCategories.CROSSING:
                    current_detections["crossings"].append((distance, direction, obj_name, conf))
                    color = (0, 255, 255)
                elif obj_name in ObjectCategories.PEDESTRIANS:
                    current_detections["pedestrians"].append((distance, direction, obj_name, conf))
                    color = (255, 0, 0)
                else:
                    continue
                
                # 绘制检测框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 使用中文绘制标签
                label = f"{display_name} {conf:.2f} {distance}"
                annotated_frame = self.chinese_drawer.put_chinese_text(
                    annotated_frame, label, (x1, y1 - 30), font_size=18, color=color
                )
                
                # 绘制方向指示
                cx = int(center_x)
                cv2.circle(annotated_frame, (cx, int(center_y)), 5, (0, 255, 0), -1)
        
        # 绘制分区线
        left_x = int(w * Config.LEFT_BOUND)
        right_x = int(w * Config.RIGHT_BOUND)
        cv2.line(annotated_frame, (left_x, 0), (left_x, h), (128, 128, 128), 1)
        cv2.line(annotated_frame, (right_x, 0), (right_x, h), (128, 128, 128), 1)
        
        self.detection_history.append(current_detections)
        
        return annotated_frame, current_detections
    
    def should_alert(self, detection_type, gap_time):
        now = time.time()
        if detection_type == "barrier":
            if now - self.last_barrier_time > gap_time:
                self.last_barrier_time = now
                return True
        elif detection_type == "crossing":
            if now - self.last_cross_time > gap_time:
                self.last_cross_time = now
                return True
        elif detection_type == "person":
            if now - self.last_person_time > gap_time:
                self.last_person_time = now
                return True
        return False
    
    def get_stable_detections(self):
        if not self.detection_history:
            return None
        
        barrier_count = 0
        crossing_count = 0
        person_count = 0
        recent_barriers = []
        recent_crossings = []
        recent_persons = []
        
        for detections in list(self.detection_history)[-3:]:
            if detections["barriers"]:
                barrier_count += 1
                recent_barriers.extend(detections["barriers"])
            if detections["crossings"]:
                crossing_count += 1
                recent_crossings.extend(detections["crossings"])
            if detections["pedestrians"]:
                person_count += 1
                recent_persons.extend(detections["pedestrians"])
        
        return {
            "barrier_stable": barrier_count >= 2,
            "crossing_stable": crossing_count >= 2,
            "person_stable": person_count >= 2,
            "recent_barriers": recent_barriers,
            "recent_crossings": recent_crossings,
            "recent_persons": recent_persons
        }
    
    def generate_alerts(self, detections):
        stable = self.get_stable_detections()
        if not stable:
            return
        
        if stable["barrier_stable"] and self.should_alert("barrier", Config.BARRIER_GAP):
            barriers = stable["recent_barriers"]
            if barriers:
                nearest = min(barriers, key=lambda x: {"近距离": 0, "中距离": 1, "远距离": 2}[x[0]])
                dist, direction, name, _ = nearest
                
                if dist == "近距离":
                    msg = f"注意！{direction}有障碍物，请立即避让"
                    self.voice.speak(msg, priority=1)
                elif dist == "中距离":
                    msg = f"{direction}检测到障碍物，请小心"
                    self.voice.speak(msg, priority=0)
                else:
                    msg = f"{direction}远处有障碍物"
                    self.voice.speak(msg, priority=0)
        
        if stable["crossing_stable"] and self.should_alert("crossing", Config.CROSS_GAP):
            crossings = stable["recent_crossings"]
            if crossings:
                dist, direction, name, _ = crossings[0]
                if dist == "近距离":
                    msg = "前方路口，请注意来往车辆，确认安全后通过"
                    self.voice.speak(msg, priority=1)
                else:
                    msg = "前方即将到达路口，注意观察"
                    self.voice.speak(msg, priority=0)
        
        if stable["person_stable"] and self.should_alert("person", Config.PERSON_GAP):
            persons = stable["recent_persons"]
            if persons:
                dist, direction, _, _ = persons[0]
                if dist == "近距离":
                    msg = f"{direction}有行人靠近，请注意"
                    self.voice.speak(msg, priority=1)
    
    def calculate_fps(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
    
    def draw_info_panel(self, frame, fps, detections):
        """绘制信息面板（中文）"""
        h, w = frame.shape[:2]
        
        # 背景条
        cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 110), (255, 255, 255), 1)
        
        # 系统信息
        runtime = time.time() - self.start_time
        info_text = f"FPS: {fps:.1f} | 运行时间: {int(runtime)}秒 | 帧数: {self.frame_count}"
        frame = self.chinese_drawer.put_chinese_text(frame, info_text, (10, 10), font_size=18, color=(0, 255, 0))
        
        # 检测统计
        barrier_count = len(detections["barriers"])
        crossing_count = len(detections["crossings"])
        person_count = len(detections["pedestrians"])
        
        status_text = f"障碍物: {barrier_count} | 路口: {crossing_count} | 行人: {person_count}"
        frame = self.chinese_drawer.put_chinese_text(frame, status_text, (10, 40), font_size=18, color=(0, 255, 255))
        
        # 图例
        legend_text = "红=障碍 黄=路口 蓝=行人"
        frame = self.chinese_drawer.put_chinese_text(frame, legend_text, (10, 70), font_size=16, color=(200, 200, 200))
        
        return frame
    
    def run(self):
        if not self.initialize():
            print("系统初始化失败，请检查依赖和环境")
            sys.exit(1)
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("[警告] 摄像头读取失败，尝试重新连接...")
                    time.sleep(1)
                    continue
                
                self.frame_count += 1
                
                annotated_frame, detections = self.process_frame(frame)
                self.generate_alerts(detections)
                fps = self.calculate_fps()
                annotated_frame = self.draw_info_panel(annotated_frame, fps, detections)
                
                # 中文窗口标题
                cv2.imshow("盲人出行辅助提醒系统 v2.0", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户请求退出")
                    break
                    
        except KeyboardInterrupt:
            print("\n收到中断信号")
        except Exception as e:
            print(f"[错误] 运行异常: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("\n" + "=" * 50)
        print("正在关闭系统...")
        
        if self.voice:
            self.voice.stop()
            print("[语音系统] 已关闭")
        
        if self.cap:
            self.cap.release()
            print("[摄像头] 已释放")
        
        cv2.destroyAllWindows()
        
        runtime = time.time() - self.start_time
        print(f"[统计] 总运行时间: {int(runtime)}秒")
        print(f"[统计] 处理帧数: {self.frame_count}")
        if runtime > 0:
            print(f"[统计] 平均FPS: {self.frame_count / runtime:.1f}")
        
        print("系统已安全关闭")
        print("=" * 50)


# ==================== 入口 ====================
if __name__ == "__main__":
    system = BlindAssistanceSystem()
    system.run()  
