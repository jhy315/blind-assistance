# -*- coding: utf-8 -*-
"""
盲人出行辅助系统 - Web版后端
功能：接收图片进行YOLO检测，返回结果
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import base64
import time
import os
import sys

app = Flask(__name__)
CORS(app)
model = None
def get_model():
    """延迟加载模型"""
    global model
    if model is None:
        import cv2
        from ultralytics import YLO
        print("[模型加载] 正在加载 YOLOv8...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
        model = YOLO(model_path)
        print("[模型加载] 成功")
    return model
# 障碍物分类
BARRIERS = {
    "car", "truck", "bus", "bicycle", "motorcycle",
    "bench", "chair", "couch", "potted plant", "fire hydrant",
    "stop sign", "parking meter", "suitcase", "backpack",
    "handbag", "sports ball", "skateboard", "surfboard",
    "dog", "cat", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}
PEDESTRIANS = {"person"}
CROSSING = {"traffic light", "stop sign"}

# 中文名称映射
CHINESE_NAMES = {
    "person": "行人", "car": "汽车", "truck": "卡车", "bus": "公交车",
    "bicycle": "自行车", "motorcycle": "摩托车", "bench": "长椅",
    "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "fire hydrant": "消防栓", "stop sign": "停车标志",
    "parking meter": "停车计时器", "traffic light": "红绿灯",
    "dog": "狗", "cat": "猫", "bird": "鸟", "horse": "马",
    "sheep": "羊", "cow": "牛", "elephant": "大象",
    "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿"
}


def estimate_distance(frame_area, box_area):
    """估算距离"""
    ratio = box_area / frame_area
    if ratio > 0.15:
        return "近距离"
    elif ratio > 0.05:
        return "中距离"
    else:
        return "远距离"


def get_direction(frame_width, center_x):
    """判断方向"""
    ratio = center_x / frame_width
    if ratio < 0.35:
        return "左侧"
    elif ratio > 0.65:
        return "右侧"
    else:
        return "正前方"


@app.route('/')
def index():
    """返回前端页面"""
    return app.send_static_file('index.html')


@app.route('/detect', methods=['POST'])
def detect():
      model = get_model()
    """接收图片，返回检测结果"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data"})
        
        # 解码base64图片
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Invalid image"})
        
        h, w = frame.shape[:2]
        frame_area = h * w
        
        # 执行检测
        results = model(frame, conf=0.45, verbose=False)
        
        detections = []
        
        for res in results:
            boxes = res.boxes
            names = res.names
            
            for box in boxes:
                cls_id = int(box.cls[0])
                obj_name = names[cls_id]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                
                distance = estimate_distance(frame_area, box_area)
                direction = get_direction(w, center_x)
                
                # 分类
                alert_type = "none"
                if obj_name in BARRIERS:
                    alert_type = "barrier"
                elif obj_name in PEDESTRIANS:
                    alert_type = "person"
                elif obj_name in CROSSING:
                    alert_type = "crossing"
                
                detections.append({
                    "name": CHINESE_NAMES.get(obj_name, obj_name),
                    "english_name": obj_name,
                    "confidence": round(conf, 2),
                    "distance": distance,
                    "direction": direction,
                    "alert_type": alert_type,
                    "bbox": [x1, y1, x2, y2]
                })
        
        return jsonify({
            "success": True,
            "detections": detections,
            "timestamp": time.time()
        })
        
    except Exception as e:
        import traceback
        print(f"[检测错误] {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == '__main__':
    import socket
    
    def get_local_ip():
        """获取本机局域网IP"""
        try:
            # 方法1：通过UDP连接获取
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            # 方法2：通过主机名获取
            return socket.gethostbyname(socket.gethostname())
    
    local_ip = get_local_ip()
    
    print("=" * 50)
    print("  盲人出行辅助系统 - Web版")
    print("=" * 50)
    print("服务器启动，访问地址：")
    print(f"  本机:     http://127.0.0.1:5000")
    print(f"  局域网:   http://{local_ip}:5000")
    print("=" * 50)
    print("手机请连接同一WiFi，访问上述局域网地址")
    print("按 CTRL+C 停止服务器")
    print("=" * 50)
    
    # 0.0.0.0 允许外部设备访问
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
