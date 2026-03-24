from flask import Flask, request, send_file
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import io
import json
import cv2
import numpy as np

app = Flask(__name__)


def region_to_polygon(region, width, height):
    """Convert region data to list of pixel (x, y) tuples.
    Supports:
      - polygon: [[x,y], ...] in 0.0–1.0          (new format, 4-6 points)
      - {x1,y1,x2,y2} in 0.0–1.0                  (legacy bbox)
      - {x,y,w,h} in 0–100                          (legacy bbox %)
    """
    if isinstance(region, list):
        return [(int(p[0] * width), int(p[1] * height)) for p in region]
    if isinstance(region, dict):
        if 'x1' in region:
            x1 = int(region['x1'] * width)
            y1 = int(region['y1'] * height)
            x2 = int(region['x2'] * width)
            y2 = int(region['y2'] * height)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        x = int(region.get('x', 0) / 100 * width)
        y = int(region.get('y', 0) / 100 * height)
        w = int(region.get('w', 10) / 100 * width)
        h = int(region.get('h', 10) / 100 * height)
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return []


def polygon_centroid(pts):
    """Return (cx, cy) centroid of a polygon."""
    cx = sum(p[0] for p in pts) // len(pts)
    cy = sum(p[1] for p in pts) // len(pts)
    return cx, cy


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}


@app.route('/convert-pdf', methods=['POST'])
def convert_pdf():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    pdf_bytes = request.files['file'].read()
    if not pdf_bytes:
        return {'error': 'Empty file'}, 400
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200, use_pdftocairo=True)
    except Exception:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200)
    if not images:
        return {'error': 'Failed to convert PDF'}, 500
    img_io = io.BytesIO()
    images[0].save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='plan.png')


@app.route('/crop-plan', methods=['POST'])
def crop_plan():
    """
    Finds the largest closed contour (apartment outline) and returns a cropped image.
    After cropping, plan occupies 100% of the frame — GPT coordinates 0.0-1.0 map to real walls.
    Input:  multipart/form-data { image: <PNG binary> }
    Output: PNG binary (cropped)
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400

    img_bytes = request.files['image'].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        return {'error': 'Failed to decode image'}, 400

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Threshold: floor plans are usually dark lines on white background
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: return original
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')

    # Pick the largest contour by area, ignoring tiny noise
    min_area = (w * h) * 0.05  # at least 5% of image
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')

    largest = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add margin (1% of each dimension)
    margin_x = max(10, int(w * 0.01))
    margin_y = max(10, int(h * 0.01))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    # Sanity check: cropped area must be at least 30% of original
    if (x2 - x1) * (y2 - y1) < w * h * 0.30:
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')

    cropped_cv = img_cv[y1:y2, x1:x2]
    _, buf = cv2.imencode('.png', cropped_cv)
    img_io = io.BytesIO(buf.tobytes())
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='cropped_plan.png')


@app.route('/annotate-rooms', methods=['POST'])
def annotate_rooms():
    """
    Draws semi-transparent room labels on the floor plan.
    Input:  multipart/form-data  { image: <PNG binary>, rooms_json: <JSON string> }
    Output: PNG binary
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    if 'rooms_json' not in request.form:
        return {'error': 'No rooms_json provided'}, 400

    img = Image.open(request.files['image']).convert('RGBA')
    rooms = json.loads(request.form['rooms_json'])
    width, height = img.size

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, room in enumerate(rooms):
        poly = region_to_polygon(room.get('polygon') or room.get('region_percent', {}), width, height)
        if not poly:
            continue

        # Semi-transparent blue fill + border
        draw.polygon(poly, fill=(59, 130, 246, 70), outline=(59, 130, 246, 200))

        # Room name label at centroid
        label = room.get('name', f'Помещение {i + 1}')
        cx, cy = polygon_centroid(poly)
        draw.text((cx, cy), label, fill=(0, 0, 100, 230))

    result = Image.alpha_composite(img, overlay).convert('RGB')
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='annotated_rooms.png')


@app.route('/annotate-changes', methods=['POST'])
def annotate_changes():
    """
    Draws wall-level annotations on the floor plan.
    Input:  multipart/form-data  { image: <PNG binary>, rooms_json: <JSON string>, changes: <JSON string> }
    Output: PNG binary
    Colors: red = illegal, yellow = requires_approval
    Each change may have wall_segments: [[x1,y1,x2,y2], ...] in 0.0-1.0 coordinates.
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    for field in ('rooms_json', 'changes'):
        if field not in request.form:
            return {'error': f'{field} is required'}, 400

    img = Image.open(request.files['image']).convert('RGBA')
    rooms_raw = json.loads(request.form['rooms_json'])
    rooms = json.loads(rooms_raw) if isinstance(rooms_raw, str) else rooms_raw
    changes_raw = json.loads(request.form['changes'])
    changes = json.loads(changes_raw) if isinstance(changes_raw, str) else changes_raw
    width, height = img.size

    line_colors = {
        'illegal':           (220, 38,  38,  255),   # red
        'requires_approval': (217, 119, 6,   255),   # amber
    }
    label_bg = {
        'illegal':           (220, 38,  38,  220),
        'requires_approval': (217, 119, 6,   220),
    }

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    badge_num = 1
    room_map = {r.get('id', ''): r for r in rooms}
    for change in changes:
        cls = change.get('classification', 'legal')
        if cls == 'legal':
            continue  # nothing to mark for legal changes

        line_color = line_colors.get(cls, (150, 150, 150, 255))
        bg_color   = label_bg.get(cls, (150, 150, 150, 220))
        line_w = max(4, int(min(width, height) * 0.008))  # ~0.8% of shorter side

        segments = change.get('wall_segments')

        # Fallback: if no wall segments, draw outline around the room polygon
        if not segments:
            room = room_map.get(change.get('room_id', ''), {})
            room_poly = region_to_polygon(
                room.get('polygon') or room.get('region_percent', {}), width, height
            )
            if room_poly:
                draw.polygon(room_poly, fill=None, outline=line_color)
                mx = sum(p[0] for p in room_poly) // len(room_poly)
                my = sum(p[1] for p in room_poly) // len(room_poly)
                r = line_w * 3
                draw.ellipse([mx - r, my - r, mx + r, my + r], fill=bg_color)
                draw.text((mx - r // 2, my - r), str(badge_num), fill=(255, 255, 255, 255))
                badge_num += 1
            continue

        drawn = False
        for seg in segments:
            if len(seg) != 4:
                continue
            sx1 = int(seg[0] * width)
            sy1 = int(seg[1] * height)
            sx2 = int(seg[2] * width)
            sy2 = int(seg[3] * height)

            # Skip degenerate segments covering >60% in both axes
            if abs(sx2 - sx1) > width * 0.6 and abs(sy2 - sy1) > height * 0.6:
                continue

            draw.line([(sx1, sy1), (sx2, sy2)], fill=line_color, width=line_w)

            # Short perpendicular ticks at endpoints (like a wall bracket)
            import math
            dx, dy = sx2 - sx1, sy2 - sy1
            length = math.hypot(dx, dy) or 1
            px, py = -dy / length, dx / length  # perpendicular unit vector
            tick = line_w * 2
            for ex, ey in [(sx1, sy1), (sx2, sy2)]:
                draw.line(
                    [(int(ex + px * tick), int(ey + py * tick)),
                     (int(ex - px * tick), int(ey - py * tick))],
                    fill=line_color, width=line_w
                )
            drawn = True

        if drawn:
            # Badge near the first valid segment midpoint
            seg0 = segments[0]
            if len(seg0) == 4:
                mx = int((seg0[0] + seg0[2]) / 2 * width)
                my = int((seg0[1] + seg0[3]) / 2 * height)
                r = line_w * 3
                draw.ellipse([mx - r, my - r, mx + r, my + r], fill=bg_color)
                draw.text((mx - r // 2, my - r), str(badge_num), fill=(255, 255, 255, 255))
            badge_num += 1

    result = Image.alpha_composite(img, overlay).convert('RGB')
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='annotated_changes.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
