from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QProgressBar, QApplication,
                            QInputDialog, QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import face_recognition
import pickle
from datetime import datetime
from database.attendance_db import AttendanceDB

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸检测系统")
        self.setup_ui()
        
        # 初始化摄像头
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 加载YOLO模型
        self.model = YOLO('yolov8n-face.pt')
        
        # 图像增强参数
        self.enable_enhancement = False

        # 添加批处理结果存储路径
        self.output_dir = Path("detection_results")
        self.output_dir.mkdir(exist_ok=True)

        # 添加人脸数据相关的属性
        self.known_face_encodings = []
        self.known_face_names = []
        self.faces_db_file = "faces_db.pkl"
        self.load_face_database()

        # 初始化数据库
        self.attendance_db = AttendanceDB()
        
        # 添加考勤记录缓存，避免重复记录
        self.attendance_cache = {}

    def setup_ui(self):
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        # 创建控制按钮
        start_button = QPushButton("开始检测")
        stop_button = QPushButton("停止检测")
        enhance_button = QPushButton("图像增强")
        
        start_button.clicked.connect(self.start_detection)
        stop_button.clicked.connect(self.stop_detection)
        enhance_button.clicked.connect(self.toggle_enhancement)
        
        right_layout.addWidget(start_button)
        right_layout.addWidget(stop_button)
        right_layout.addWidget(enhance_button)
        
        # 在right_layout中添加新的按钮和进度条
        batch_button = QPushButton("批量处理图片")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        batch_button.clicked.connect(self.process_batch_images)
        
        right_layout.addWidget(batch_button)
        right_layout.addWidget(self.progress_bar)
        right_layout.addStretch()
        
        # 添加人脸注册按钮
        register_face_button = QPushButton("注册新面孔")
        register_face_button.clicked.connect(self.register_new_face)
        right_layout.addWidget(register_face_button)
        
        # 添加考勤记录查看按钮
        view_records_button = QPushButton("查看考勤记录")
        view_records_button.clicked.connect(self.show_attendance_records)
        right_layout.addWidget(view_records_button)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        central_widget.setLayout(main_layout)

    def start_detection(self):
        self.timer.start(30)  # 30ms 刷新率

    def stop_detection(self):
        self.timer.stop()

    def toggle_enhancement(self):
        self.enable_enhancement = not self.enable_enhancement

    def enhance_image(self, frame):
        # 图像增强处理
        # 提高亮度和对比度
        alpha = 1.2  # 对比度
        beta = 10    # 亮度
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # 使用更快速的降噪方法
        # enhanced = cv2.fastNlMeansDenoisingColored(enhanced)  # 删除这个耗时的操作
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)  # 使用高斯模糊替代，速度更快
        
        # 可选：添加锐化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            if self.enable_enhancement:
                frame = self.enhance_image(frame)
            
            current_time = datetime.now()
            
            # 运行YOLO检测
            results = self.model(frame, conf=0.5)
            
            # 转换为RGB进行人脸识别
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 在图像上绘制检测结果
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 提取人脸区域进行识别
                    face_encoding = face_recognition.face_encodings(rgb_frame, 
                        [(y1, x2, y2, x1)])[0]
                    
                    # 与已知人脸比对
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.6)
                    
                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                        
                        # 检查是否需要记录考勤
                        if name not in self.attendance_cache or \
                           (current_time - self.attendance_cache[name]).seconds > 300:  # 5分钟内不重复记录
                            self.attendance_db.add_record(name)
                            self.attendance_cache[name] = current_time
                    
                    # 绘制边框和姓名
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 转换图像格式并显示
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.camera.release() 

    def process_batch_images(self):
        """批量处理图片文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder_path:
            return

        # 获取所有支持的图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in Path(folder_path).glob('*') 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            return

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(image_files))
        self.progress_bar.setValue(0)

        # 处理每张图片
        for i, image_path in enumerate(image_files):
            # 读取图片
            frame = cv2.imread(str(image_path))
            
            if self.enable_enhancement:
                frame = self.enhance_image(frame)
            
            # 运行YOLO检测
            results = self.model(frame, conf=0.5)
            
            # 在图像上绘制检测结果
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 保存处理后的图片
            output_path = self.output_dir / f"detected_{image_path.name}"
            cv2.imwrite(str(output_path), frame)
            
            # 更新进度条
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()  # 保持UI响应

        # 处理完成后隐藏进度条
        self.progress_bar.setVisible(False)

    def process_single_image(self, image_path):
        """处理单张图片"""
        frame = cv2.imread(str(image_path))
        if frame is None:
            return None
            
        if self.enable_enhancement:
            frame = self.enhance_image(frame)
        
        # 运行YOLO检测
        results = self.model(frame, conf=0.5)
        
        # 在图像上绘制检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

    def load_face_database(self):
        """加载已保存的人脸数据"""
        try:
            with open(self.faces_db_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
        except FileNotFoundError:
            # 如果文件不存在，使用空列表
            pass

    def save_face_database(self):
        """保存人脸数据到文件"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.faces_db_file, 'wb') as f:
            pickle.dump(data, f)

    def register_new_face(self):
        """注册新的人脸"""
        # 暂停视频更新
        self.timer.stop()
        
        # 获取当前帧
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # 检测人脸
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            self.timer.start(30)
            return
        
        # 获取人脸编码
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        # 获取姓名
        name, ok = QInputDialog.getText(self, '注册新面孔', '请输入姓名:')
        if ok and name:
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.save_face_database()
        
        # 恢复视频更新
        self.timer.start(30)

    def show_attendance_records(self):
        """显示考勤记录"""
        records_window = QWidget()
        records_window.setWindowTitle("考勤记录")
        layout = QVBoxLayout()
        
        # 创建表格
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["姓名", "时间", "状态"])
        
        # 获取今日记录
        records = self.attendance_db.get_today_records()
        table.setRowCount(len(records))
        
        # 填充数据
        for i, (name, timestamp, status) in enumerate(records):
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(timestamp))
            table.setItem(i, 2, QTableWidgetItem(status))
        
        layout.addWidget(table)
        records_window.setLayout(layout)
        records_window.resize(400, 300)
        records_window.show()
        
        # 保持窗口引用，防止被垃圾回收
        self._records_window = records_window