from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

class YOLOv8Detector:
    def __init__(self, model_path="D:\\yolov8\\蓝云数据迁移\\yolov8\\yolov8\\yolov8\\runs\\detect\\exp_LSKA-SENet-WIoU2\\weights\\best.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, image_bytes):
        # 将字节流转换为图像
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        
        # 进行推理
        results = self.model(image)
        
        # 处理结果
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()
                class_name = result.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
        
        # 生成带标注的图像
        annotated_image = results[0].plot()
        
        return detections, annotated_image