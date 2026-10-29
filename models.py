"""
models.py - Database Layer for VisionVoice AI
Covers Syllabus:
- Software Architecture, Design, and Patterns (OOP, Data Access Object pattern)
- Database schema Design (SQLite relational tables for Image Audits & Settings)
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")


class DatabaseManager:
    """Singleton Data Access Object (DAO) for image recognition logs and settings."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create tables for image recognition audits and optional API settings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_recognitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    top_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    caption TEXT NOT NULL,
                    dominant_hex TEXT DEFAULT '#38bdf8',
                    engine_used TEXT DEFAULT 'MobileNetV2 CNN',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Auto-migrate column if older table exists
            try:
                cursor.execute("ALTER TABLE image_recognitions ADD COLUMN engine_used TEXT DEFAULT 'MobileNetV2 CNN'")
            except Exception:
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            conn.commit()

    def log_recognition(self, source_name: str, top_label: str, confidence: float, caption: str, dominant_hex: str, engine_used: str = "MobileNetV2 CNN") -> int:
        """Store image recognition result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO image_recognitions (source_name, top_label, confidence, caption, dominant_hex, engine_used)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_name, top_label, confidence, caption, dominant_hex, engine_used)
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_recognitions(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieve recent recognition audits."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM image_recognitions ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear_history(self):
        """Clear audit history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM image_recognitions")
            conn.commit()

    def set_setting(self, key: str, value: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    def get_setting(self, key: str) -> Optional[str]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate metrics for presentation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total_scans, AVG(confidence) as avg_conf FROM image_recognitions")
            row = cursor.fetchone()
            return {
                "total_scans": row["total_scans"] or 0,
                "avg_confidence": round(row["avg_conf"] or 0.0, 1)
            }


db = DatabaseManager()
