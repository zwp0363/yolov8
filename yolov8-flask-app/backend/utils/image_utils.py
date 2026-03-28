import base64
import cv2
import numpy as np

def hex_to_image(hex_str):
    """将十六进制字符串转换为图像"""
    img_bytes = bytes.fromhex(hex_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def image_to_base64(image):
    """将图像转换为 base64 编码"""
    _, img_encoded = cv2.imencode('.jpg', image)
    return base64.b64encode(img_encoded).decode('utf-8')