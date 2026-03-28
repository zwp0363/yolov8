from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import cv2
import numpy as np
import io
import sqlite3
import json
from datetime import datetime
from models.detector import YOLOv8Detector
import os

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化数据库
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    
    # 创建表（如果不存在）
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  original_image BLOB,
                  annotated_image BLOB,
                  detections TEXT)''')
    
    # 检查是否存在elapsed_time列
    c.execute("PRAGMA table_info(detections)")
    columns = [column[1] for column in c.fetchall()]
    
    # 如果不存在elapsed_time列，添加它
    if 'elapsed_time' not in columns:
        c.execute("ALTER TABLE detections ADD COLUMN elapsed_time REAL")
    
    # 如果不存在model_path列，添加它
    if 'model_path' not in columns:
        c.execute("ALTER TABLE detections ADD COLUMN model_path TEXT")
    
    conn.commit()
    conn.close()

# 确保模型目录存在
os.makedirs('models', exist_ok=True)

init_db()

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    # 处理模型
    model_path = None
    
    # 检查是否上传了自定义模型
    if 'custom_model' in request.files:
        custom_model = request.files['custom_model']
        if custom_model.filename.endswith('.pt'):
            # 保存自定义模型
            model_filename = os.path.join('models', custom_model.filename)
            custom_model.save(model_filename)
            model_path = os.path.abspath(model_filename)
    else:
        # 使用预定义模型
        model_path = request.form.get('model_path', "D:\\yolov8\\蓝云数据迁移\\yolov8\\yolov8\\yolov8\\runs\\detect\\exp_LSKA-SENet-WIoU2\\weights\\best.pt")
    
    # 初始化检测器（使用指定的模型）
    detector = YOLOv8Detector(model_path)
    
    # 进行检测
    detections, annotated_image, elapsed_time, model_path = detector.detect(image_bytes)
    
    # 将标注后的图像转换为字节流
    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    img_bytes = img_encoded.tobytes()
    
    # 保存到数据库
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("INSERT INTO detections (timestamp, elapsed_time, model_path, original_image, annotated_image, detections) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               elapsed_time,
               model_path,
               image_bytes,
               img_bytes,
               json.dumps(detections)))
    conn.commit()
    detection_id = c.lastrowid
    conn.close()
    
    # 返回检测结果和标注后的图像
    return jsonify({
        "detections": detections,
        "image": img_bytes.hex(),  # 将图像转换为十六进制字符串
        "id": detection_id,
        "elapsed_time": elapsed_time,
        "model_path": model_path
    })

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/result/<int:id>')
def result(id):
    return send_file('static/result.html')

@app.route('/history')
def history():
    return send_file('static/history.html')

@app.route('/api/history')
def get_history():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT id, timestamp, elapsed_time, model_path, detections FROM detections ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "elapsed_time": row[2],
            "model_path": row[3],
            "detections": json.loads(row[4])
        })
    
    return jsonify(history)

@app.route('/api/result/<int:id>')
def get_result(id):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, elapsed_time, model_path, annotated_image, detections FROM detections WHERE id = ?", (id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return jsonify({"error": "Result not found"}), 404
    
    return jsonify({
        "timestamp": row[0],
        "elapsed_time": row[1],
        "model_path": row[2],
        "image": row[3].hex(),
        "detections": json.loads(row[4])
    })

@app.route('/api/delete/<int:id>', methods=['DELETE'])
def delete_result(id):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("DELETE FROM detections WHERE id = ?", (id,))
    conn.commit()
    affected_rows = c.rowcount
    conn.close()
    
    if affected_rows == 0:
        return jsonify({"error": "Result not found"}), 404
    
    return jsonify({"success": True, "message": "检测记录已删除"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)