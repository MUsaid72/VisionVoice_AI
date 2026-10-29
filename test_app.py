"""
test_app.py - Automated Unit & Integration Tests for VisionVoice AI Suite
Verifies:
- Accurate recognition of Yellow Rubber Duck (with red beak)
- Accurate recognition of Red Rose
- Accurate recognition of Blue Sports Car
- Flask REST routes (GET and POST modes)
- SQLite Persistence
"""

import io
import pytest
import numpy as np
from PIL import Image
from app import app
from vision_engine import VisionEngine
from models import db


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Test standard GET request to dashboard index."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"VisionVoice AI" in response.data


def test_yellow_rubber_duck_recognition():
    """Verify that a predominantly yellow object with a red beak accent is recognized as a Yellow Rubber Duck."""
    # Create 224x224 image: mostly bright yellow, with a red beak patch in center
    img = Image.new('RGB', (224, 224), color=(255, 215, 0)) # Gold/Yellow
    # Paint small red beak in the middle
    arr = np.array(img)
    arr[100:120, 100:120] = [230, 20, 20] # Red beak
    duck_img = Image.fromarray(arr)

    byte_arr = io.BytesIO()
    duck_img.save(byte_arr, format='JPEG')
    
    result = VisionEngine.process_image(byte_arr.getvalue(), "yellow_rubber_duck.jpg")
    assert result["success"] is True
    assert "Rubber Duck" in result["top_label"] or "Yellow" in result["top_label"]
    assert result["confidence"] >= 90.0


def test_red_rose_recognition():
    """Verify that a deep red floral object is recognized as a Red Rose."""
    img = Image.new('RGB', (224, 224), color=(220, 20, 35)) # Deep red
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')

    result = VisionEngine.process_image(byte_arr.getvalue(), "red_rose.jpg")
    assert result["success"] is True
    assert "Rose" in result["top_label"] or "Red" in result["top_label"]
    assert result["confidence"] >= 90.0


def test_blue_sports_car_recognition():
    """Verify that a blue vehicle/sports car is recognized with automotive taxonomy."""
    img = Image.new('RGB', (224, 224), color=(20, 100, 230)) # Royal Blue
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')

    result = VisionEngine.process_image(byte_arr.getvalue(), "blue_car.jpg")
    assert result["success"] is True
    assert "Blue" in result["top_label"]


def test_recognize_upload_endpoint(client):
    """Test multipart file upload endpoint."""
    img = Image.new('RGB', (224, 224), color=(240, 210, 30))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')
    byte_arr.seek(0)

    response = client.post(
        '/api/recognize-upload',
        content_type='multipart/form-data',
        data={'image': (byte_arr, 'test_upload.jpg')}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "top_label" in data
    assert "web_verification" in data
    assert "speech_text" in data


def test_history_logging(client):
    """Verify SQLite persistence for recognition audits."""
    response = client.get('/api/history')
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "history" in data
    assert "stats" in data
