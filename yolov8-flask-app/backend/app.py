from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
from models.detector import YOLOv8Detector

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化检测器
detector = YOLOv8Detector()

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    # 进行检测
    detections, annotated_image = detector.detect(image_bytes)
    
    # 将标注后的图像转换为字节流
    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    img_bytes = img_encoded.tobytes()
    
    # 返回检测结果和标注后的图像
    return jsonify({
        "detections": detections,
        "image": img_bytes.hex()  # 将图像转换为十六进制字符串
    })

@app.route('/')
def index():
    return send_file('static/index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)