from flask import Flask, request, send_file, jsonify
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont
import os
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


def order_points(pts):
    """ Упорядочивает точки: [top-left, top-right, bottom-right, bottom-left] """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


@app.route('/crop-plan', methods=['POST'])
def crop_plan():
    try:
        # 1. Получаем координаты углов из Query-параметров
        corners_raw = request.args.get('corners')
        if not corners_raw:
            return json.dumps({"error": "No corners coordinates provided in query"}), 400, {'Content-Type': 'application/json'}

        corners_data = json.loads(corners_raw)
        # Ожидаем формат [{"x":... , "y":...}, ...]

        # 2. Получаем файл изображения
        if 'image' not in request.files:
            return json.dumps({"error": "No image file provided"}), 400, {'Content-Type': 'application/json'}

        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h_img, w_img = img.shape[:2]

        # 3. Преобразуем проценты ИИ в пиксели
        pts = []
        for c in corners_data:
            pts.append([float(c['x']) * w_img / 100, float(c['y']) * h_img / 100])

        pts = np.array(pts, dtype="float32")
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # 4. Вычисляем ширину и высоту нового изображения
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # 5. Выполняем Perspective Warp (выравнивание)
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, m, (max_width, max_height))

        # 6. Возвращаем результат как файл
        _, buffer = cv2.imencode('.jpg', warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        io_buf = io.BytesIO(buffer)

        return send_file(io_buf, mimetype='image/jpeg', as_attachment=True, download_name='cropped_plan.jpg')

    except Exception as e:
        return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}


def find_rooms_geometric(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    h_img, w_img = img.shape[:2]

    # 1. Улучшаем контраст (CLAHE) — вытягивает бледные линии на фото
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Бинаризация с уменьшенным окном (11 вместо 21) — ловит тонкие линии
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Убираем мелкий шум, но сохраняем стены
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Склеиваем стены (Dilation) — замыкает прерывистые линии на фото
    dilate_kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(opening, dilate_kernel, iterations=2)

    # 5. Поиск контуров
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_rooms = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # Порог 0.5% — ловит маленькие санузлы
        if area > (h_img * w_img * 0.005):
            x, y, w, h = cv2.boundingRect(cnt)

            detected_rooms.append({
                "internal_id": f"room_{i}",
                "bbox": {
                    "x1": round((x / w_img) * 100, 1),
                    "y1": round((y / h_img) * 100, 1),
                    "x2": round(((x + w) / w_img) * 100, 1),
                    "y2": round(((y + h) / h_img) * 100, 1)
                },
                "area_pct": round((area / (h_img * w_img)) * 100, 2)
            })

    # Сортировка (сверху вниз)
    detected_rooms.sort(key=lambda r: r['bbox']['y1'])
    return detected_rooms


@app.route('/detect-rooms', methods=['POST'])
def detect_rooms():
    """
    Geometrically detects rooms in a floor plan image.
    Input:  multipart/form-data { image: <binary> }
    Output: JSON { detected_rooms: [...] }
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400

    image_bytes = request.files['image'].read()
    rooms = find_rooms_geometric(image_bytes)
    return json.dumps({"detected_rooms": rooms}), 200, {'Content-Type': 'application/json'}


def process_full_photo(img, ai_json):
    h_img, w_img = img.shape[:2]

    # 1. Находим лист бумаги на столе
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_paper = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    contours_paper, _ = cv2.findContours(thresh_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_paper:
        return ai_json  # Если лист не найден, возвращаем как есть

    # Берем самый большой контур (лист БТИ)
    paper_cnt = max(contours_paper, key=cv2.contourArea)
    px, py, pw, ph = cv2.boundingRect(paper_cnt)

    # 2. Ищем комнаты ТОЛЬКО внутри области листа
    roi = gray[py:py + ph, px:px + pw]
    thresh_rooms = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # "Жирним" стены внутри листа
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh_rooms, kernel, iterations=1)
    contours_rooms, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    geo_rooms = []
    for cnt in contours_rooms:
        if cv2.contourArea(cnt) > (pw * ph * 0.01):  # Минимум 1% от площади листа
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Переводим координаты в глобальные координаты всего фото (%)
            geo_rooms.append({
                'x1': ((px + rx) / w_img) * 100,
                'y1': ((py + ry) / h_img) * 100,
                'x2': ((px + rx + rw) / w_img) * 100,
                'y2': ((py + ry + rh) / h_img) * 100
            })

    # 3. Приземляем точки ИИ в найденные гео-комнаты
    for shot in ai_json.get('shots', []):
        raw_x = shot.get('x', 50)
        raw_y = shot.get('y', 50)

        for room in geo_rooms:
            if room['x1'] <= raw_x <= room['x2'] and room['y1'] <= raw_y <= room['y2']:
                margin_w = (room['x2'] - room['x1']) * 0.15
                margin_h = (room['y2'] - room['y1']) * 0.15

                if shot.get('position') == "центр":
                    shot['x'] = round(room['x1'] + (room['x2'] - room['x1']) / 2, 1)
                    shot['y'] = round(room['y1'] + (room['y2'] - room['y1']) / 2, 1)
                else:
                    # Сдвигаем точку от стены, если она слишком близко
                    shot['x'] = round(max(room['x1'] + margin_w, min(raw_x, room['x2'] - margin_w)), 1)
                    shot['y'] = round(max(room['y1'] + margin_h, min(raw_y, room['y2'] - margin_h)), 1)
                break

    return ai_json


def get_font(size):
    """Ищем доступный кириллический шрифт на сервере"""
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text_pil(img_pil, text, position, color=(255, 0, 0), font_size=20):
    """Отрисовка текста через Pillow на PIL изображении"""
    draw = ImageDraw.Draw(img_pil)
    font = get_font(font_size)
    draw.text(position, text, font=font, fill=color)
    return img_pil


@app.route('/process-shots', methods=['POST'])
def process_shots():
    """
    Draws semi-transparent shot points with numbered labels on the image.
    Input:  multipart/form-data { image: <binary> } + query param ai_data=<JSON {rooms, shots}>
    Output: JPEG with annotated shot positions
    """
    try:
        # 1. Получаем JSON и картинку
        ai_data_raw = request.args.get('ai_data')
        if not ai_data_raw:
            return json.dumps({"error": "No ai_data in URL"}), 400, {'Content-Type': 'application/json'}
        data = json.loads(ai_data_raw)
        if isinstance(data, list):
            data = data[0]

        file = request.files.get('image')
        if not file:
            return json.dumps({"error": "No image in body"}), 400, {'Content-Type': 'application/json'}
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return json.dumps({"error": "Failed to decode image"}), 400, {'Content-Type': 'application/json'}
        h, w = img.shape[:2]

        # Конвертируем OpenCV (BGR) в PIL (RGB) для работы с прозрачностью
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Создаем прозрачный слой для точек
        overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        blue_color_trans = (0, 0, 255, 128)   # 50% прозрачности
        blue_color_text  = (0, 0, 255, 255)   # Непрозрачный синий для текста

        # 2. Отрисовка точек и текста
        for index, shot in enumerate(data.get('shots', []), start=1):
            sx, sy = int(shot['x'] * w / 100), int(shot['y'] * h / 100)

            # Полупрозрачный синий круг на оверлее
            radius = 20
            draw_overlay.ellipse(
                [sx - radius, sy - radius, sx + radius, sy + radius],
                fill=blue_color_trans
            )

            # Белая цифра (порядок фото) внутри круга
            draw_text_pil(pil_img, str(index), (sx - 7, sy - 12), color=(255, 255, 255, 255), font_size=20)

            # Синяя надпись позиции рядом с точкой
            label = shot.get('pos', '')
            draw_text_pil(pil_img, label, (sx + radius + 5, sy - 10), color=blue_color_text, font_size=18)

        # Комбинируем оригинальное изображение с прозрачным оверлеем
        final_img_pil = Image.alpha_composite(pil_img.convert('RGBA'), overlay)

        # 3. Конвертируем обратно в OpenCV и выводим
        final_img_cv2 = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGBA2BGR)
        _, buffer = cv2.imencode('.jpg', final_img_cv2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

    except Exception as e:
        return json.dumps({"error": str(e)}), 500, {'Content-Type': 'application/json'}


def process_plan(img, data):
    h_img, w_img = img.shape[:2]
    shots = data.get('shots', [])
    detected_rooms = data.get('detected_rooms', [])

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_num = get_font(24)
    font_txt = get_font(18)

    BLUE_FILL  = (0, 0, 255, 130)
    BLUE_SOLID = (0, 0, 255, 255)
    WHITE      = (255, 255, 255, 255)

    for i, shot in enumerate(shots, start=1):
        # Берем координаты из shot или из detected_rooms по индексу
        if 'x' in shot and 'y' in shot:
            sx, sy = int(shot['x'] * w_img / 100), int(shot['y'] * h_img / 100)
        elif i - 1 < len(detected_rooms):
            room = detected_rooms[i - 1]
            bbox = room['bbox']

            if shot.get('position') == "в центре":
                sx = int((bbox['x1'] + bbox['x2']) / 2 * w_img / 100)
                sy = int((bbox['y1'] + bbox['y2']) / 2 * h_img / 100)
            else:
                # "В углу" или "у стены" — с отступом 20%
                sx = int((bbox['x1'] + (bbox['x2'] - bbox['x1']) * 0.2) * w_img / 100)
                sy = int((bbox['y1'] + (bbox['y2'] - bbox['y1']) * 0.2) * h_img / 100)
        else:
            continue

        r = 25
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=BLUE_FILL, outline=WHITE, width=2)
        draw.text((sx - 8, sy - 15), str(i), font=font_num, fill=WHITE)

        label = f"{shot.get('room_name', '')} {shot.get('position', '')}".strip()
        draw.text((sx + r + 10, sy - 10), label, font=font_txt, fill=BLUE_SOLID)

    return Image.alpha_composite(pil_img, overlay).convert('RGB')


@app.route('/process', methods=['POST'])
def handle():
    """
    Draws numbered shot points on the image using shots + detected_rooms from ai_data.
    Input:  multipart/form-data { image: <binary> } + query param ai_data=<JSON {shots, detected_rooms}>
    Output: JPEG with annotated shot positions
    """
    try:
        ai_data_raw = request.args.get('ai_data')
        if not ai_data_raw:
            return jsonify({"error": "No ai_data"}), 400

        data = json.loads(ai_data_raw)

        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No image"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        result_img = process_plan(img, data)

        buf = io.BytesIO()
        result_img.save(buf, 'JPEG', quality=95)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect-rooms-with-shots', methods=['POST'])
def detect_rooms_with_shots():
    """
    Finds paper sheet, detects rooms inside it, then snaps AI shot points into safe positions.
    Input:  multipart/form-data { image: <binary> } + query param ai_data=<JSON>
    Output: JSON (updated ai_json with corrected shot coordinates)
    """
    ai_data_raw = request.args.get('ai_data')
    if not ai_data_raw:
        return json.dumps({"error": "No ai_data query param provided"}), 400, {'Content-Type': 'application/json'}

    ai_json = json.loads(ai_data_raw)

    if 'image' not in request.files:
        return json.dumps({"error": "No image file provided"}), 400, {'Content-Type': 'application/json'}

    img_array = np.frombuffer(request.files['image'].read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return json.dumps({"error": "Failed to decode image"}), 400, {'Content-Type': 'application/json'}

    result = process_full_photo(img, ai_json)
    return json.dumps(result), 200, {'Content-Type': 'application/json'}


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
