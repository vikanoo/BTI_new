from flask import Flask, request, send_file
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import io
import json
import math
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


def preprocess_for_hough(img_cv):
    """Preprocess image to enhance wall lines for Hough detection.
    Morphological closing fills small gaps in wall lines and suppresses
    isolated thin features (dimension arrows, hatching, text strokes).
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed


def find_wall_between_centroids(img_cv, c1, c2, strip_fraction=0.30, min_cos_perp=0.5):
    """Find the wall between two rooms using centroid-based strip search.

    Algorithm:
      1. C1→C2 defines the axis between room centres.
      2. Search all Hough segments whose midpoint lies within a strip of
         width ±(strip_fraction * |C1C2|) around the C1→C2 line, and
         whose midpoint projection falls between C1 and C2.
      3. Among candidates, keep only those roughly perpendicular to C1→C2
         (cos of angle with perpendicular direction ≥ min_cos_perp).
      4. Score by: distance of midpoint from mid(C1,C2) + perpendicularity penalty.
         Return the best-scoring segment.

    c1, c2: pixel (x, y) centroids.
    Returns (x1, y1, x2, y2) or None.
    """
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    L = math.hypot(dx, dy)
    if L < 1:
        return None

    ux, uy = dx / L, dy / L   # unit vector C1→C2
    strip_w = L * strip_fraction

    binary = preprocess_for_hough(img_cv)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=40,
                             minLineLength=30, maxLineGap=15)
    if lines is None:
        return None

    best_line = None
    best_score = 0.0  # best = longest segment passing all filters

    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        vmx, vmy = mx - c1[0], my - c1[1]

        # Perpendicular distance from midpoint to the infinite C1→C2 line
        dist_perp = abs(vmx * uy - vmy * ux)
        if dist_perp > strip_w:
            continue

        # Projection of midpoint along C1→C2; must lie between C1 and C2
        proj_along = vmx * ux + vmy * uy
        if proj_along < -L * 0.1 or proj_along > L * 1.1:
            continue

        # Check that this Hough segment is roughly perpendicular to C1→C2.
        # Perpendicular direction to C1→C2 is (-uy, ux).
        hdx, hdy = x2 - x1, y2 - y1
        hL = math.hypot(hdx, hdy)
        if hL < 1:
            continue
        cos_perp = abs(hdx / hL * (-uy) + hdy / hL * ux)
        if cos_perp < min_cos_perp:
            continue

        # Key filter: C1 and C2 must be on OPPOSITE sides of this line.
        # The shared wall between two rooms always separates their centroids.
        # Normal to the Hough line: (-hdy/hL, hdx/hL)
        nx, ny = -hdy / hL, hdx / hL
        sign1 = (c1[0] - mx) * nx + (c1[1] - my) * ny
        sign2 = (c2[0] - mx) * nx + (c2[1] - my) * ny
        if sign1 * sign2 >= 0:   # same side → not the wall between these rooms
            continue

        # Score: prefer longer segments (longer = more wall-like, less noise)
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len > best_score:
            best_score = seg_len
            best_line = (x1, y1, x2, y2)

    return best_line


def find_longest_hough_in_bbox(img_cv, poly, width, height, margin=0.20):
    """Find the longest Hough line segment within the bounding box of a room polygon.
    Used for single-room changes where we don't have a second centroid to guide search.

    poly:   list of pixel (x, y) tuples defining the room polygon
    margin: fractional expansion of the bounding box on each side
    Returns (x1, y1, x2, y2) or None.
    """
    if not poly:
        return None

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx1_f, by1_f = min(xs), min(ys)
    bx2_f, by2_f = max(xs), max(ys)

    mw = (bx2_f - bx1_f) * margin
    mh = (by2_f - by1_f) * margin
    bx1 = max(0, int(bx1_f - mw))
    by1 = max(0, int(by1_f - mh))
    bx2 = min(width,  int(bx2_f + mw))
    by2 = min(height, int(by2_f + mh))

    binary = preprocess_for_hough(img_cv)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=40,
                             minLineLength=30, maxLineGap=15)
    if lines is None:
        return None

    best_line = None
    best_len = 0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if bx1 <= mx <= bx2 and by1 <= my <= by2:
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len > best_len:
                best_len = seg_len
                best_line = (x1, y1, x2, y2)

    return best_line



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


def perspective_transform(image, corners_pct):
    h_img, w_img = image.shape[:2]

    # Преобразуем проценты от ИИ в пиксели
    pts1 = np.float32([
        [c['x'] * w_img / 100, c['y'] * h_img / 100] for c in corners_pct
    ])

    # Вычисляем размеры нового изображения
    width = int(max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3])))
    height = int(max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2])))

    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Матрица трансформации
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (width, height))
    return result


@app.route('/crop-plan', methods=['POST'])
def crop_plan():
    """
    Applies perspective transform to straighten a floor plan photo.
    Input:  multipart/form-data { image: <binary>, corners: <JSON array of 4 {x, y} in percent> }
    Output: JPEG binary (straightened)
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    if 'corners' not in request.form:
        return {'error': 'No corners provided'}, 400

    img_bytes = request.files['image'].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        return {'error': 'Failed to decode image'}, 400

    corners = json.loads(request.form['corners'])
    final_img = perspective_transform(img_cv, corners)

    _, buf = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    img_io = io.BytesIO(buf.tobytes())
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg', download_name='cropped_plan.jpg')


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

        cx, cy = polygon_centroid(poly)
        r = max(12, int(min(width, height) * 0.018))

        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(59, 130, 246, 220))
        num = str(i + 1)
        draw.text((cx - r // 2, cy - r // 2), num, fill=(255, 255, 255, 255))

        label = room.get('name', f'Помещение {i + 1}')
        draw.text((cx + r + 4, cy - r // 2), label, fill=(0, 0, 100, 230))

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

    Strategy per change:
      - 2 rooms:  centroid-strip Hough search → line or badge-only
      - 1 room:   longest Hough line inside room bbox → line or badge-only
      - No polygon outline fallback — wrong outline is worse than no outline.
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    for field in ('rooms_json', 'changes'):
        if field not in request.form:
            return {'error': f'{field} is required'}, 400

    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
            continue

        line_color = line_colors.get(cls, (150, 150, 150, 255))
        bg_color   = label_bg.get(cls, (150, 150, 150, 220))
        line_w = max(4, int(min(width, height) * 0.008))

        affected_ids = change.get('affected_room_ids') or [change.get('room_id', '')]
        change_type  = change.get('type', '')

        # Types that involve a physical wall: Hough search makes sense.
        # Internal changes (fixtures, doorways, other) → badge-only, no wall search.
        WALL_TYPES = {'wall_removal', 'wall_addition', 'room_merge', 'room_split'}
        needs_wall_search = change_type in WALL_TYPES

        drawn_segment = None  # (x1, y1, x2, y2) pixels
        badge_pos = None

        if needs_wall_search and len(affected_ids) >= 2:
            # Wall between two rooms: centroid-strip Hough search
            r1 = room_map.get(affected_ids[0], {})
            r2 = room_map.get(affected_ids[1], {})
            poly1 = region_to_polygon(r1.get('polygon') or r1.get('region_percent', {}), width, height)
            poly2 = region_to_polygon(r2.get('polygon') or r2.get('region_percent', {}), width, height)
            if poly1 and poly2:
                c1 = polygon_centroid(poly1)
                c2 = polygon_centroid(poly2)
                drawn_segment = find_wall_between_centroids(img_cv, c1, c2)
                if badge_pos is None:
                    badge_pos = c1

        elif needs_wall_search and len(affected_ids) == 1:
            # Single-room wall change: find longest wall line inside room bbox
            room = room_map.get(affected_ids[0], {})
            poly = region_to_polygon(
                room.get('polygon') or room.get('region_percent', {}), width, height
            )
            if poly:
                drawn_segment = find_longest_hough_in_bbox(img_cv, poly, width, height)
                badge_pos = polygon_centroid(poly)

        if drawn_segment is not None:
            x1s, y1s, x2s, y2s = drawn_segment
            draw.line([(x1s, y1s), (x2s, y2s)], fill=line_color, width=line_w * 2)
            badge_pos = ((x1s + x2s) // 2, (y1s + y2s) // 2)

        # Fallback: badge_pos may still be None if polygon lookup failed for all rooms.
        # Find any available room polygon and place badge at its centroid.
        if badge_pos is None:
            for rid in affected_ids:
                room = room_map.get(rid, {})
                poly = region_to_polygon(
                    room.get('polygon') or room.get('region_percent', {}), width, height
                )
                if poly:
                    badge_pos = polygon_centroid(poly)
                    break

        if badge_pos:
            mx, my = badge_pos
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
