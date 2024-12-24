import sqlite3
from datetime import datetime
from pathlib import Path

class AttendanceDB:
    def __init__(self):
        self.db_file = Path("attendance.db")
        self.init_database()

    def init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(str(self.db_file))
        cursor = conn.cursor()
        
        # 创建考勤记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            status TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()

    def add_record(self, name, status="签到"):
        """添加考勤记录"""
        conn = sqlite3.connect(str(self.db_file))
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO attendance_records (name, timestamp, status) VALUES (?, ?, ?)",
            (name, timestamp, status)
        )
        
        conn.commit()
        conn.close()

    def get_today_records(self):
        """获取今日考勤记录"""
        conn = sqlite3.connect(str(self.db_file))
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            "SELECT name, timestamp, status FROM attendance_records WHERE date(timestamp) = ?",
            (today,)
        )
        records = cursor.fetchall()
        
        conn.close()
        return records

    def get_person_records(self, name):
        """获取指定人员的考勤记录"""
        conn = sqlite3.connect(str(self.db_file))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT timestamp, status FROM attendance_records WHERE name = ? ORDER BY timestamp DESC",
            (name,)
        )
        records = cursor.fetchall()
        
        conn.close()
        return records 