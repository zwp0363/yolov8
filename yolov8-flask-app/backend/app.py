from flask import Flask, request, jsonify, send_file, render_template, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import io
import sqlite3
import json
from datetime import datetime
from models.detector import YOLOv8Detector
import os
import hashlib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于会话管理
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
    
    # 如果不存在user_id列，添加它
    if 'user_id' not in columns:
        c.execute("ALTER TABLE detections ADD COLUMN user_id INTEGER")
    
    # 创建用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  name TEXT,
                  gender TEXT,
                  email TEXT,
                  phone TEXT,
                  avatar BLOB)''')
    
    conn.commit()
    conn.close()

# 确保模型目录存在
os.makedirs('models', exist_ok=True)

init_db()

# 密码加密函数
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 验证用户登录
def verify_user(username, password):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and user[1] == hash_password(password):
        return user[0]
    return None

# 注册新用户
def register_user(username, password, name=None, gender=None, email=None, phone=None):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    
    try:
        c.execute("INSERT INTO users (username, password, name, gender, email, phone) VALUES (?, ?, ?, ?, ?, ?)",
                  (username, hash_password(password), name, gender, email, phone))
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

# 获取用户信息
def get_user_info(user_id):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT id, username, name, gender, email, phone, avatar FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'username': user[1],
            'name': user[2],
            'gender': user[3],
            'email': user[4],
            'phone': user[5],
            'avatar': user[6]
        }
    return None

# 更新用户信息
def update_user_info(user_id, name=None, gender=None, email=None, phone=None, avatar=None):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    
    update_fields = []
    update_values = []
    
    if name:
        update_fields.append("name = ?")
        update_values.append(name)
    if gender:
        update_fields.append("gender = ?")
        update_values.append(gender)
    if email:
        update_fields.append("email = ?")
        update_values.append(email)
    if phone:
        update_fields.append("phone = ?")
        update_values.append(phone)
    if avatar:
        update_fields.append("avatar = ?")
        update_values.append(avatar)
    
    if update_fields:
        update_query = "UPDATE users SET " + ", ".join(update_fields) + " WHERE id = ?"
        update_values.append(user_id)
        c.execute(update_query, update_values)
        conn.commit()
    
    conn.close()
    return True

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    name = data.get('name')
    gender = data.get('gender')
    email = data.get('email')
    phone = data.get('phone')
    
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    
    user_id = register_user(username, password, name, gender, email, phone)
    if user_id:
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({"success": True, "user_id": user_id, "username": username})
    else:
        return jsonify({"error": "用户名已存在"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user_id = verify_user(username, password)
    if user_id:
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({"success": True, "user_id": user_id, "username": username})
    else:
        return jsonify({"error": "用户名或密码错误"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/user/info', methods=['GET'])
def get_user():
    if 'user_id' not in session:
        return jsonify({"error": "未登录"}), 401
    
    user_info = get_user_info(session['user_id'])
    if user_info:
        # 处理头像
        if user_info['avatar']:
            user_info['avatar'] = user_info['avatar'].hex()
        return jsonify(user_info)
    else:
        return jsonify({"error": "用户不存在"}), 404

@app.route('/api/user/update', methods=['POST'])
def update_user():
    if 'user_id' not in session:
        return jsonify({"error": "未登录"}), 401
    
    data = request.form
    name = data.get('name')
    gender = data.get('gender')
    email = data.get('email')
    phone = data.get('phone')
    avatar = request.files.get('avatar')
    
    avatar_bytes = None
    if avatar:
        avatar_bytes = avatar.read()
    
    update_user_info(session['user_id'], name, gender, email, phone, avatar_bytes)
    return jsonify({"success": True})

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
    
    # 获取用户ID（如果已登录）
    user_id = session.get('user_id')
    
    c.execute("INSERT INTO detections (timestamp, elapsed_time, model_path, user_id, original_image, annotated_image, detections) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               elapsed_time,
               model_path,
               user_id,
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

@app.route('/api/history')
def get_history():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    
    # 如果用户已登录，只返回用户自己的检测记录
    if 'user_id' in session:
        c.execute("SELECT id, timestamp, elapsed_time, model_path, detections FROM detections WHERE user_id = ? ORDER BY id DESC", (session['user_id'],))
    else:
        c.execute("SELECT id, timestamp, elapsed_time, model_path, detections FROM detections WHERE user_id IS NULL ORDER BY id DESC")
    
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
    
    # 检查记录是否属于当前用户
    if 'user_id' in session:
        c.execute("SELECT timestamp, elapsed_time, model_path, annotated_image, detections, user_id FROM detections WHERE id = ?", (id,))
        row = c.fetchone()
        if row and row[5] != session['user_id']:
            return jsonify({"error": "无权访问此记录"}), 403
    else:
        c.execute("SELECT timestamp, elapsed_time, model_path, annotated_image, detections, user_id FROM detections WHERE id = ? AND user_id IS NULL", (id,))
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
    
    # 检查记录是否属于当前用户
    if 'user_id' in session:
        c.execute("SELECT user_id FROM detections WHERE id = ?", (id,))
        row = c.fetchone()
        if row and row[0] != session['user_id']:
            conn.close()
            return jsonify({"error": "无权删除此记录"}), 403
    else:
        c.execute("DELETE FROM detections WHERE id = ? AND user_id IS NULL", (id,))
    conn.commit()
    affected_rows = c.rowcount
    conn.close()
    
    if affected_rows == 0:
        return jsonify({"error": "Result not found"}), 404
    
    return jsonify({"success": True, "message": "检测记录已删除"})

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/result/<int:id>')
def result(id):
    return send_file('static/result.html')

@app.route('/history')
def history():
    return send_file('static/history.html')

@app.route('/login')
def login_page():
    return send_file('static/login.html')

@app.route('/register')
def register_page():
    return send_file('static/register.html')

@app.route('/profile')
def profile_page():
    return send_file('static/profile.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)