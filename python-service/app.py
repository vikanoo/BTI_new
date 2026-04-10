from flask import Flask, request, send_file, jsonify
import requests
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont
import os
import io
import json
import math
import cv2
import numpy as np
import uuid
import re
import base64
import time
from datetime import datetime
from io import BytesIO
from supabase import create_client

app = Flask(__name__)

# Инициализация клиента OpenAI
# Рекомендуется задать переменную окружения: export OPENAI_API_KEY='ваш_ключ'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ВАШ_КЛЮЧ_ОТ_OPENAI")
VALID_TOKEN = os.getenv("BTI_SERVICE_TOKEN", "bti_secure_token_2026")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
client = OpenAI(api_key=OPENAI_API_KEY)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None



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


# =========================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# =========================
def load_image(url):
    import requests as req
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = req.get(url, headers=headers)

    if resp.status_code != 200:
        raise Exception("Ошибка загрузки изображения")

    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Ошибка декодирования изображения")

    return img


# =========================
# PREPROCESS
# =========================
def _preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return edges


# =========================
# СТЕНЫ (УЛУЧШЕНО)
# =========================
def _build_walls(edges):
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=4)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed


# =========================
# WATERSHED
# =========================
def _segment_rooms(walls):
    inv = cv2.bitwise_not(walls)
    h, w = inv.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, mask, (0, 0), 0)

    dist = cv2.distanceTransform(flood, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist, 0.18, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    sure_bg = cv2.dilate(flood, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(walls, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    return markers


# =========================
# ИЗВЛЕЧЕНИЕ КОМНАТ
# =========================
def _extract_rooms(markers, img):
    h, w = img.shape[:2]
    rooms = []

    for label in np.unique(markers):
        if label <= 1:
            continue

        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > (w * h) * 0.5:
                continue
            if area < (w * h) * 0.003:
                continue

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            polygon = [
                {"x": float(p[0][0] / w), "y": float(p[0][1] / h)}
                for p in approx
            ]

            rooms.append({
                "id": f"room_{len(rooms)+1}",
                "polygon": polygon,
                "center": {"x": float(cx / w), "y": float(cy / h)},
                "area_pct": float(area / (w * h) * 100)
            })

    return rooms


# =========================
# FALLBACK (если плохо нашло)
# =========================
def _fallback_segmentation(edges, img):
    h, w = img.shape[:2]
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rooms = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h) * 0.01:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        rooms.append({
            "id": f"room_{len(rooms)+1}",
            "bbox": {
                "x1": x / w,
                "y1": y / h,
                "x2": (x + cw) / w,
                "y2": (y + ch) / h
            }
        })

    return rooms


# =========================
# DEBUG
# =========================
def _debug_draw(img, rooms):
    debug = img.copy()
    h, w = img.shape[:2]

    for r in rooms:
        pts = np.array([
            [int(p["x"] * w), int(p["y"] * h)]
            for p in r.get("polygon", [])
        ], np.int32)

        if len(pts) > 0:
            cv2.polylines(debug, [pts], True, (0, 255, 0), 2)

        if "center" in r:
            cx = int(r["center"]["x"] * w)
            cy = int(r["center"]["y"] * h)
            cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imwrite("debug.png", debug)


# =========================
# /detect-rooms-url
# =========================
@app.route('/detect-rooms-url', methods=['POST'])
def detect_rooms_url():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    url = data.get("planUrl")
    if not url:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = load_image(url)

        edges = _preprocess(img)
        walls = _build_walls(edges)
        markers = _segment_rooms(walls)
        rooms = _extract_rooms(markers, img)

        if len(rooms) < 2:
            rooms = _fallback_segmentation(edges, img)

        _debug_draw(img, rooms)

        return jsonify({"rooms": rooms, "count": len(rooms)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_bti_plan(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Failed to decode image"}

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    min_area_percent = 0.01
    total_area = h * w

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < (total_area * min_area_percent):
            continue

        epsilon = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        polygon_norm = [{"x": round(pt[0][0] / w, 5), "y": round(pt[0][1] / h, 5)} for pt in approx]
        center_norm = {"x": round(cx / w, 5), "y": round(cy / h, 5)}

        rooms.append({
            "id": f"room_{len(rooms) + 1}",
            "area_px": area,
            "polygon": polygon_norm,
            "center": center_norm
        })

    return {"rooms": rooms, "info": {"original_width": w, "original_height": h}}


@app.route('/extract-rooms', methods=['POST'])
def handle_extract_rooms():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()

    result = process_bti_plan(image_bytes)
    return jsonify(result)


def _preprocess_image_shots(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph


def _get_rooms_data(processed_image):
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rooms = []
    if hierarchy is None:
        return rooms
    h, w = processed_image.shape[:2]
    min_area = (h * w) * 0.005
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hierarchy[0][i][3] != -1 and area > min_area:
            rect = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else rect[0] + rect[2] // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else rect[1] + rect[3] // 2
            rooms.append({"bbox": rect, "center": (cx, cy)})
    rooms.sort(key=lambda r: (r["center"][1], r["center"][0]))
    return rooms


def _calculate_point(description, room):
    x, y, w, h = room["bbox"]
    cx, cy = room["center"]
    desc = description.lower()
    if "окно" in desc:
        return (cx, y + int(h * 0.15))
    if "двер" in desc or "вход" in desc:
        return (cx, y + int(h * 0.85))
    if "угол" in desc or "углу" in desc:
        return (x + int(w * 0.2), y + int(h * 0.2))
    return (cx, cy)


@app.route('/draw-shots', methods=['POST'])
def handle_draw_shots():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400

        shots_json_raw = request.form.get('shots_json')
        if not shots_json_raw:
            return jsonify({"error": "No shots_json in body"}), 400

        shots_data = json.loads(shots_json_raw)

        img_array = np.frombuffer(request.files['image'].read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        processed = _preprocess_image_shots(image)
        rooms = _get_rooms_data(processed)

        for shot in shots_data:
            try:
                r_idx = int(shot.get("room_id", "room_1").split("_")[1]) - 1
                if 0 <= r_idx < len(rooms):
                    px, py = _calculate_point(shot.get("position", ""), rooms[r_idx])
                    cv2.circle(image, (px, py), 12, (0, 0, 255), -1)
                    cv2.circle(image, (px, py), 14, (255, 255, 255), 2)
                    cv2.putText(image, shot.get("shot_id", ""), (px + 15, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except:
                continue

        _, buffer = cv2.imencode('.png', image)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


class BTIPlanAnalyzer:
    """Анализатор планов БТИ для расстановки точек съёмки"""

    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Не удалось загрузить изображение")
        self.height, self.width = self.original.shape[:2]
        self.min_room_area = int(self.width * self.height * 0.001)

    def analyze(self, output_image_path=None):
        binary = self._preprocess()
        rooms = self._find_rooms(binary)
        rooms_with_points = []
        for room in rooms:
            room['shooting_points'] = self._generate_shooting_points(room)
            rooms_with_points.append(room)
        result = self._build_result(rooms_with_points)
        if output_image_path:
            self._draw_annotations(rooms_with_points, output_image_path)
            result['annotated_image_path'] = output_image_path
        return result

    def _preprocess(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _find_rooms(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rooms = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_room_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.width * 0.9 and h > self.height * 0.9:
                continue
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            room_type = self._detect_room_type(x, y, w, h)
            area_m2 = round(area / 1000, 1)
            rooms.append({
                'id': f"room_{len(rooms)+1}",
                'name': self._get_room_name(room_type, area_m2),
                'type': room_type,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area_pixels': int(area),
                'area_estimate_m2': area_m2
            })
        rooms.sort(key=lambda r: r['area_pixels'], reverse=True)
        return rooms

    def _detect_room_type(self, x, y, w, h):
        roi = self.original[y:y+h, x:x+w]
        if roi.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        blue_ratio = np.sum(blue_mask > 0) / roi.size
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 50, 255]))
        gray_ratio = np.sum(gray_mask > 0) / roi.size
        aspect_ratio = w / h
        is_narrow = aspect_ratio < 0.4 or aspect_ratio > 2.5
        if blue_ratio > 0.05:
            return "bathroom"
        elif gray_ratio > 0.08:
            return "kitchen"
        elif is_narrow and w * h < 50000:
            return "corridor"
        elif w * h < 25000:
            return "storage"
        return "room"

    def _get_room_name(self, room_type, area_m2):
        names = {
            'room': f"Жилая комната {area_m2} м²",
            'kitchen': f"Кухня {area_m2} м²",
            'bathroom': f"Санузел {area_m2} м²",
            'corridor': f"Коридор {area_m2} м²",
            'storage': f"Кладовая {area_m2} м²",
            'unknown': f"Помещение {area_m2} м²"
        }
        return names.get(room_type, f"Помещение {area_m2} м²")

    def _generate_shooting_points(self, room):
        x, y, w, h = room['bbox']
        points = [
            {
                'position': (x + w // 2, y + h - 20),
                'direction': 'на дальнюю стену и противоположный угол',
                'instruction': 'Захватить всё помещение с порога, показать глубину'
            },
            {
                'position': (x + w - 20, y + 20),
                'direction': 'на вход и центр помещения',
                'instruction': 'Снять помещение из дальнего угла, показать обратную перспективу'
            }
        ]
        if w * h > 50000:
            points.append({
                'position': (x + 20, y + h - 20),
                'direction': 'на центр комнаты',
                'instruction': 'Захватить угол комнаты и основное пространство'
            })
        return points[:3]

    def _build_result(self, rooms_with_points):
        result = {'rooms': [], 'shots': [], 'total_rooms': len(rooms_with_points), 'total_shots': 0}
        shot_counter = 1
        for room in rooms_with_points:
            result['rooms'].append({
                'id': room['id'],
                'name': room['name'],
                'type': room['type'],
                'area_estimate_m2': room['area_estimate_m2'],
                'center': {'x': room['center'][0], 'y': room['center'][1]}
            })
            for point in room['shooting_points']:
                result['shots'].append({
                    'shot_id': f"shot_{shot_counter}",
                    'room_id': room['id'],
                    'room_name': room['name'],
                    'position': {'x': point['position'][0], 'y': point['position'][1]},
                    'direction': point['direction'],
                    'instruction': point['instruction']
                })
                shot_counter += 1
        result['total_shots'] = shot_counter - 1
        return result

    def _draw_annotations(self, rooms_with_points, output_path):
        img_pil = Image.fromarray(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = font_small = ImageFont.load_default()
        colors = {
            'room': (255, 0, 0),
            'kitchen': (0, 255, 0),
            'bathroom': (128, 0, 128),
            'corridor': (0, 0, 255),
            'storage': (255, 165, 0),
            'unknown': (128, 128, 128)
        }
        shot_counter = 1
        for room in rooms_with_points:
            color = colors.get(room['type'], (0, 0, 0))
            x, y, w, h = room['bbox']
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            draw.text((x+5, y+5), room['name'], fill=color, font=font_small)
            for point in room['shooting_points']:
                px, py = point['position']
                draw.ellipse([px-8, py-8, px+8, py+8], outline=color, fill=(255, 255, 255), width=2)
                draw.text((px-4, py-6), str(shot_counter), fill=color, font=font)
                shot_counter += 1
        legend_x = self.width - 180
        legend_y = 10
        draw.text((legend_x, legend_y), "ЛЕГЕНДА:", fill=(0, 0, 0), font=font)
        legend_y += 20
        type_labels = {
            'room': 'Жилые комнаты', 'kitchen': 'Кухни',
            'bathroom': 'Санузлы', 'corridor': 'Коридоры', 'storage': 'Кладовки'
        }
        for type_name, label in type_labels.items():
            draw.text((legend_x, legend_y), f"• {label}", fill=colors.get(type_name, (0, 0, 0)), font=font_small)
            legend_y += 16
        img_pil.save(output_path)


def analyze_bti_plan(image_data, save_annotated=True, output_dir="/tmp"):
    temp_file = None
    input_path = None
    request_id = str(uuid.uuid4())[:8]
    try:
        if isinstance(image_data, bytes):
            input_path = os.path.join(output_dir, f"bti_{request_id}.jpg")
            temp_file = input_path
            with open(input_path, 'wb') as f:
                f.write(image_data)
        elif isinstance(image_data, str) and os.path.exists(image_data):
            input_path = image_data
        else:
            raise ValueError("image_data должен быть bytes или путём к файлу")
        analyzer = BTIPlanAnalyzer(input_path)
        output_image = None
        if save_annotated:
            output_image = os.path.join(output_dir, f"bti_annotated_{request_id}.jpg")
        result = analyzer.analyze(output_image_path=output_image)
        if output_image:
            result['annotated_image_path'] = output_image
        return result
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def process_bti_shots_request(input_data):
    try:
        if isinstance(input_data, str):
            data_list = json.loads(input_data)
        else:
            data_list = input_data

        if not data_list or len(data_list) == 0:
            return {'status': 'error', 'error': 'Empty data array'}

        item = data_list[0]
        chat_id = item.get('chatId')
        rooms = item.get('rooms_json', [])
        shots = item.get('shots_json', [])
        image_base64 = item.get('data', '')

        if not image_base64:
            return {'status': 'error', 'error': 'No image data found'}
        if not shots:
            return {'status': 'error', 'error': 'No shots data found'}

        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        image_bytes = base64.b64decode(image_base64)
        img_pil = Image.open(BytesIO(image_bytes))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        draw = ImageDraw.Draw(img_pil)
        width, height = img_pil.size

        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ]
            font = font_small = None
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, 16)
                        font_small = ImageFont.truetype(fp, 12)
                        break
                    except Exception:
                        continue
            if font is None:
                font = font_small = ImageFont.load_default()

        rooms_dict = {room.get('id'): room for room in rooms if room.get('id')}
        colors = {
            'identified': (34, 139, 34),
            'visual_only': (255, 140, 0),
            'default': (220, 20, 60)
        }

        def parse_position(position_text, room_id=None):
            if not position_text:
                return (width // 2, height // 2)
            text = position_text.lower()
            coord_match = re.search(r'\(?(\d+)[,\s]+(\d+)\)?', text)
            if coord_match:
                return (int(coord_match.group(1)), int(coord_match.group(2)))
            cx, cy = width // 2, height // 2
            if any(w in text for w in ['центр', 'center']):
                return (cx, cy)
            elif any(w in text for w in ['угол', 'corner']):
                return (width - 50, 50)
            elif any(w in text for w in ['вход', 'дверь', 'door']):
                return (cx, height - 30)
            elif any(w in text for w in ['окно', 'window']):
                return (width - 50, cy)
            elif any(w in text for w in ['плита', 'stove']):
                return (width - 100, cy)
            elif any(w in text for w in ['ванна', 'bath']):
                return (width - 80, height - 100)
            return (cx, cy)

        for room in rooms:
            if 'bbox' in room and len(room['bbox']) == 4:
                x1, y1, x2, y2 = room['bbox']
                color = colors.get(room.get('type', 'default'), colors['default'])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 5, y1 + 5), room.get('name', room.get('id', '')), fill=color, font=font_small)

        drawn_shots = []
        for i, shot in enumerate(shots, 1):
            shot_id = shot.get('shot_id', f'shot_{i}')
            room_id = shot.get('room_id')
            x, y = parse_position(shot.get('position', 'центр'), room_id)
            x = max(20, min(x, width - 20))
            y = max(20, min(y, height - 20))
            color = colors.get(rooms_dict.get(room_id, {}).get('type', 'default'), colors['default'])
            radius = 12
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, fill=(255, 255, 255), width=3)
            draw.text((x - 5, y - 7), str(i), fill=color, font=font)
            drawn_shots.append({
                'shot_id': shot_id, 'number': i,
                'position': {'x': x, 'y': y}, 'room_id': room_id,
                'room_name': shot.get('room_name', ''),
                'direction': shot.get('direction', ''),
                'instruction': shot.get('instruction', '')
            })

        legend_x, legend_y = width - 200, 10
        draw.text((legend_x, legend_y), "ЛЕГЕНДА:", fill=(0, 0, 0), font=font)
        legend_y += 22
        for _, color, label in [
            ('identified', (34, 139, 34), 'identified (известные)'),
            ('visual_only', (255, 140, 0), 'visual_only (визуально)')
        ]:
            draw.ellipse([legend_x + 5, legend_y + 2, legend_x + 15, legend_y + 12], fill=color, outline=color)
            draw.text((legend_x + 20, legend_y), label, fill=(0, 0, 0), font=font_small)
            legend_y += 18
        draw.text((legend_x, legend_y), f"Всего точек: {len(drawn_shots)}", fill=(100, 100, 100), font=font_small)

        output_buffer = BytesIO()
        img_pil.save(output_buffer, format='PNG')
        result_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        return {
            'status': 'success', 'chat_id': chat_id,
            'annotated_image_base64': result_base64,
            'shots_count': len(drawn_shots), 'drawn_shots': drawn_shots,
            'message': f'Нанесено {len(drawn_shots)} точек съёмки'
        }

    except Exception as e:
        return {'status': 'error', 'error': str(e), 'message': f'Ошибка обработки: {str(e)}'}



@app.route('/apply-grid', methods=['POST'])
def apply_beacons():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400

    file = request.files['file']
    # Оптимальный шаг для маяков — около 80-100 пикселей
    step = int(request.args.get('step', 100))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image: {str(e)}"}, 400

    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Настройка шрифта для номеров точек
    font_size = 14
    try:
        font = ImageFont.load_default()
    except:
        font = None

    radius = 12  # Размер красного кружка
    counter = 1

    # Рисуем сетку из точек
    # Начинаем с отступа step, чтобы точки не липли к краям
    for y in range(step, height - 20, step):
        for x in range(step, width - 20, step):
            # 1. Рисуем подложку (белый ореол), чтобы точку было видно на темном фоне
            draw.ellipse([x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2], fill=(255, 255, 255))

            # 2. Рисуем саму красную точку
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 0, 0))

            # 3. Пишем номер точки белым цветом по центру
            # Немного корректируем координаты текста, чтобы он был по центру круга
            text_str = str(counter)
            text_pos = (x - 6, y - 7) if len(text_str) == 1 else (x - 10, y - 7)

            draw.text(text_pos, text_str, fill=(255, 255, 255), font=font)

            counter += 1

    # Сохраняем результат
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

# def process_image(file_storage):
#     """Оптимизация изображения для распознавания мелких деталей и дробей."""
#     img = Image.open(file_storage)
#     if img.mode in ("RGBA", "P"):
#         img = img.convert("RGB")
#     # Высокое разрешение для распознавания мелких цифр (санузлы, балконы)
#     img.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
#     buffer = io.BytesIO()
#     img.save(buffer, format="JPEG", quality=95)
#     return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_image_embedding(image_bytes):
    """Генерирует текстовый эмбеддинг изображения через OpenAI (описание + embed)."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    desc_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Опиши кратко структуру плана БТИ: список помещений, их площади и расположение."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}}
        ]}],
        max_tokens=300
    )
    description = desc_resp.choices[0].message.content

    embed_resp = client.embeddings.create(model="text-embedding-3-small", input=description)
    return embed_resp.data[0].embedding


def get_best_example(image_bytes):
    """Находит ближайший пример БТИ в Supabase по векторному сходству."""
    if not supabase:
        return None
    try:
        query_vector = get_image_embedding(image_bytes)
        res = supabase.rpc("match_bti_examples", {
            "query_embedding": query_vector,
            "match_threshold": 0.5,
            "match_count": 1
        }).execute()
        if res.data:
            return res.data[0]['example_json']
    except Exception:
        pass
    return None


def step_1_ocr_analysis(image_base64, target_area=None, example_json=None):
    """Этап 1: Распознавание БТИ с ПРЯМЫМ ЗАПРЕТОМ на догадки"""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    area_hint = f"Общая площадь: {target_area} м2." if target_area else ""

    rag_instruction = ""
    if example_json:
        rag_instruction = f"\nОРИЕНТИРУЙСЯ НА ЭТОТ ЭТАЛОН (RAG):\n{json.dumps(example_json, ensure_ascii=False)}\n"

    prompt = (
        "Ты — строгий технический контролер. Твоя первая и главная задача: определить, является ли фото ПЛАНОМ БТИ (чертежом).\n\n"
        f"{rag_instruction}"
        "ПРИЗНАКИ ПЛАНА (должны быть все): \n"
        "1. Схематичные черные линии стен на светлом фоне.\n"
        "2. Технические цифры (площади, размеры).\n"
        "3. Отсутствие реальных объектов (людей, животных, мебели как на фото, офисных кресел).\n\n"
        "КРИТИЧЕСКОЕ ПРАВИЛО:\n"
        "- Если на фото изображено живое существо, мебель в пространстве, интерьер или просто случайный предмет — это НЕ ПЛАН.\n"
        "- В этом случае СТОП. Верни только: {'is_plan': false, 'error_message': 'На фото изображен объект или интерьер, а не технический план БТИ. Пожалуйста, загрузите чертеж.'}\n\n"
        "ТОЛЬКО ЕСЛИ ЭТО ПЛАН:\n"
        "- Перечисли комнаты. Названия — только по тексту. Если текста нет — 'Помещение'.\n"
        "- Площадь: только из центра комнат. ИГНОРИРУЙ размеры вдоль стен (это длины).\n"
        "Верни JSON: {'is_plan': true, 'rooms': [{'id': '...', 'name': '...', 'area': float_or_null}]}"
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}}
        ]}],
        "response_format": {"type": "json_object"}
    }
    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return res.json()['choices'][0]['message']['content']


def step_2_photo_planning(ocr_json):
    """Этап 2: Строгое дополнение существующего JSON точками съемки"""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    prompt = (
        f"У тебя есть JSON данные: {ocr_json}\n\n"
        "Твоя единственная задача: ДОБАВИТЬ в каждое помещение список точек съемки.\n"
        "СТРОГИЕ ПРАВИЛА:\n"
        "1. СТРУКТУРА: Не меняй структуру JSON. Не переименовывай 'rooms' в 'съемка'. Не удаляй 'id', 'name', 'area'.\n"
        "2. ДОПОЛНЕНИЕ: Внутри каждого объекта в массиве 'rooms' создай ключ 'photo_points'.\n"
        "3. СОДЕРЖИМОЕ 'photo_points': Сгенерируй от 2 до 4 объектов с ключами 'location' и 'view'.\n"
        "4. ЯЗЫК: Значения для 'location' и 'view' пиши СТРОГО НА РУССКОМ.\n"
        "5. ОГРАНИЧЕНИЯ: Используй только слова: дверной проем, угол слева/справа от входа, центр стены, окно, противоположная стена.\n"
        "6. ЗАПРЕТЫ: Никакой мебели, никакой атмосферы, никаких 360 градусов.\n\n"
        "Верни ВЕСЬ входящий JSON, в котором у каждой комнаты появилось поле 'photo_points'."
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return res.json()['choices'][0]['message']['content']


@app.route('/analyze-bti', methods=['POST'])
def bti_endpoint():
    # 1. Токен
    if request.args.get('token', '').strip() != VALID_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Площадь
    raw_area = request.args.get('total_area', '')
    target_area = None
    if raw_area and str(raw_area).lower().strip() not in ['none', 'null', 'undefined', '']:
        try:
            target_area = float(str(raw_area).replace(',', '.'))
        except Exception:
            target_area = None

    try:
        # 3. Файл (внутри try чтобы поймать Werkzeug BadRequest)
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        file_bytes = file.read()

        # RAG: ищем ближайший пример в базе знаний
        knowledge_example = get_best_example(file_bytes)

        image_base64 = base64.b64encode(file_bytes).decode('utf-8')

        # ВЫПОЛНЕНИЕ ЦЕПОЧКИ
        raw_step1 = step_1_ocr_analysis(image_base64, target_area, knowledge_example)
        step1_data = json.loads(raw_step1)

        if not step1_data.get('is_plan', True):
            return jsonify({"status": "error", "message": step1_data.get('error_message')}), 200

        raw_step2 = step_2_photo_planning(raw_step1)
        final_data = json.loads(raw_step2)

        ai_sum = sum([float(r.get('area') or 0) for r in final_data.get('rooms', [])])

        return jsonify({
            "status": "success",
            "analysis": final_data,
            "verification": {
                "calculated_sum": round(ai_sum, 2),
                "expected": target_area,
                "difference": round(abs(ai_sum - (target_area or 0)), 2) if target_area else 0
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_clip_512_embedding_hf(image_bytes, hf_token):
    """Получает вектор 512 через Hugging Face API с переданным токеном"""
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
    headers = {"Authorization": f"Bearer {hf_token}"}

    for _ in range(3):
        try:
            response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                print("Модель HF загружается, ждем 5 секунд...")
                time.sleep(5)
                continue
            else:
                raise Exception(f"Hugging Face API error: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к HF: {e}")
            time.sleep(2)

    raise Exception("Hugging Face API недоступно или токен невалиден")


@app.route('/add-to-rag', methods=['POST'])
def add_to_rag():
    if not HUGGINGFACE_TOKEN:
        return jsonify({"error": "HUGGINGFACE_TOKEN not configured"}), 500

    data = request.json
    image_url = data.get('image_url')
    raw_response = data.get('confirmed_json')

    if not image_url or not raw_response:
        return jsonify({"error": "image_url and confirmed_json are required"}), 400

    try:
        # 1. Приведение к формату РАГ
        if isinstance(raw_response, list) and len(raw_response) > 0:
            raw_data = raw_response[0]
        else:
            raw_data = raw_response

        if raw_data.get("status") == "success":
            rag_data = raw_data.get("analysis", {})
        elif raw_data.get("status") == "error":
            rag_data = {
                "is_plan": False,
                "error_message": raw_data.get("message", "Не является планом")
            }
        else:
            rag_data = raw_data

        # 2. Скачивание изображения
        img_response = requests.get(image_url, timeout=15)
        img_response.raise_for_status()

        # 3. Генерация эмбеддинга 512 через HF API
        embedding = get_clip_512_embedding_hf(img_response.content, HUGGINGFACE_TOKEN)

        # 4. Запись в Supabase
        new_row = {
            "image_path": image_url,
            "example_json": rag_data,
            "embedding": embedding
        }

        result = supabase.table("bti_examples").insert(new_row).execute()

        return jsonify({
            "status": "success",
            "added_format": rag_data,
            "id": result.data[0]['id'] if result.data else None,
            "dimensions": len(embedding)
        }), 200

    except Exception as e:
        print(f"RAG Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
