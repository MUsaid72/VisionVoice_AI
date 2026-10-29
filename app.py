"""
app.py - Main Flask Web Application for VisionVoice AI Suite
Covers Syllabus:
- Developing AI Applications with Python and Flask
- Request and Response Objects (GET/POST modes), File Uploads, Dynamic Routes
- Deep Learning & Computer Vision REST Service
"""

import os
import secrets
import socket
import ipaddress
import urllib.parse
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
from PIL import Image
from models import db
from vision_engine import VisionEngine

# Prevent Decompression Bomb DoS attacks (OWASP A04: Insecure Design)
Image.MAX_IMAGE_PIXELS = 10_000_000

app = Flask(__name__, static_folder="static", template_folder="templates")

# Secure Secret Key loaded from environment variable or securely generated at startup
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max payload limit


# ==========================================
# SECURITY HEADERS MIDDLEWARE
# ==========================================
@app.after_request
def apply_security_headers(response):
    """Enforce defense-in-depth security response headers (OWASP A05)."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "img-src 'self' data: https:; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' https:;"
    )
    return response


# ==========================================
# DEFENSIVE UTILITIES (SSRF PREVENTION)
# ==========================================
def is_safe_remote_url(url: str) -> bool:
    """
    Prevents Server-Side Request Forgery (SSRF - OWASP A10).
    Blocks loopback (127.0.0.1), private subnets (10.0.0.0/8, 192.168.0.0/16, 172.16.0.0/12),
    and cloud metadata endpoints (169.254.169.254).
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        
        # In isolated environments, allow well-known HTTPS domains if unresolvable locally
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except socket.gaierror:
            pass  # Fall through if DNS check occurs inside an isolated local network
        return True
    except Exception:
        return False


# ==========================================
# ERROR HANDLING
# ==========================================
@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"success": False, "error": str(e), "status": 400}), 400

@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "error": f"API endpoint not found: {request.path}", "status": 404}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def handle_server_error(e):
    return jsonify({"success": False, "error": "Internal Server Error: " + str(e), "status": 500}), 500


# ==========================================
# ROUTES
# ==========================================
@app.route('/', methods=['GET'])
def index():
    """Render the VisionVoice AI multimodal dashboard."""
    stats = db.get_stats()
    return render_template('index.html', stats=stats)


@app.route('/api/recognize-upload', methods=['POST'])
def recognize_upload():
    """Classify local uploaded image via Multi-Layer Vision Engine."""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No image file selected"}), 400

    # Sanitize filename (OWASP A03: Path Traversal prevention)
    safe_name = secure_filename(file.filename) or "uploaded_asset.jpg"

    # Enforce file extension whitelist
    allowed_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in allowed_exts:
        return jsonify({"success": False, "error": f"Unsupported format '{ext}'. Allowed: {', '.join(allowed_exts)}"}), 400

    img_bytes = file.read()
    gemini_key = db.get_setting("gemini_api_key") or os.getenv("GEMINI_API_KEY")

    result = VisionEngine.process_image(img_bytes, source_name=safe_name, gemini_api_key=gemini_key)

    if result.get("success"):
        db.log_recognition(
            source_name=safe_name,
            top_label=result["top_label"],
            confidence=result["confidence"],
            caption=result["caption"],
            dominant_hex=result["properties"]["dominant_hex"],
            engine_used=result.get("engine_used", "MobileNetV2 CNN")
        )

    return jsonify(result)


@app.route('/api/recognize-url', methods=['POST'])
def recognize_url():
    """Download and classify an image from a direct image URL with SSRF protection."""
    data = request.get_json() or {}
    url = data.get('url', '').strip()

    if not url:
        return jsonify({"success": False, "error": "Image URL is required"}), 400

    # SSRF Protection (OWASP A10)
    if not is_safe_remote_url(url):
        return jsonify({
            "success": False,
            "error": "Access Denied: URL targets internal network, loopback, or invalid protocol (SSRF Protection)."
        }), 403

    try:
        import requests
        headers = {"User-Agent": "VisionVoiceAI-Client/2.0"}

        # Stream download with 16MB ceiling to prevent Memory Exhaustion DoS
        max_bytes = 16 * 1024 * 1024
        downloaded = bytearray()

        with requests.get(url, headers=headers, timeout=8, stream=True) as resp:
            resp.raise_for_status()

            # Verify Content-Type header if provided
            content_type = resp.headers.get('Content-Type', '').lower()
            if content_type and not any(t in content_type for t in ['image/', 'application/octet-stream']):
                return jsonify({"success": False, "error": f"Target URL returned non-image Content-Type: {content_type}"}), 400

            for chunk in resp.iter_content(chunk_size=65536):
                downloaded.extend(chunk)
                if len(downloaded) > max_bytes:
                    return jsonify({"success": False, "error": "Image exceeds maximum allowed size (16MB)."}), 413

        img_bytes = bytes(downloaded)
        source_name = secure_filename(url.split('/')[-1]) or "remote_image.jpg"
        gemini_key = db.get_setting("gemini_api_key") or os.getenv("GEMINI_API_KEY")

        result = VisionEngine.process_image(img_bytes, source_name=source_name, gemini_api_key=gemini_key)

        if result.get("success"):
            db.log_recognition(
                source_name=source_name,
                top_label=result["top_label"],
                confidence=result["confidence"],
                caption=result["caption"],
                dominant_hex=result["properties"]["dominant_hex"],
                engine_used=result.get("engine_used", "MobileNetV2 CNN")
            )

        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to download image from URL: {str(e)}"}), 400


@app.route('/api/history', methods=['GET'])
def get_history():
    """Retrieve audit history from SQLite."""
    history = db.get_recent_recognitions(limit=15)
    stats = db.get_stats()
    return jsonify({
        "success": True,
        "history": history,
        "stats": stats
    })


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear recognition audit logs."""
    db.clear_history()
    return jsonify({"success": True, "message": "History cleared successfully."})


@app.route('/api/settings/api-key', methods=['GET', 'POST'])
def api_key_settings():
    """Save or check optional Google Gemini Vision API key."""
    if request.method == 'POST':
        data = request.get_json() or {}
        key = data.get('api_key', '').strip()
        db.set_setting("gemini_api_key", key)
        return jsonify({"success": True, "message": "Gemini Vision key updated."})
    else:
        current = db.get_setting("gemini_api_key") or os.getenv("GEMINI_API_KEY") or ""
        masked = f"{current[:4]}...{current[-4:]}" if len(current) > 8 else ("Active" if current else "None")
        return jsonify({"success": True, "has_key": bool(current), "masked_key": masked})


if __name__ == '__main__':
    # Determine environment
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1')
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))

    print("==================================================")
    print(f"VisionVoice AI - Running Server on http://{host}:{port}")
    print(f"Debug Mode: {debug_mode} (Keep False in Production)")
    print("==================================================")
    app.run(host=host, port=port, debug=debug_mode)
