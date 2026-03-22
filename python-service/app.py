from flask import Flask, request, send_file
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import io
import json

app = Flask(__name__)


def region_to_pixels(region, width, height):
    """Convert region_percent (x, y, w, h as 0–100) to pixel box."""
    x = int(region.get('x', 0) / 100 * width)
    y = int(region.get('y', 0) / 100 * height)
    w = int(region.get('w', 10) / 100 * width)
    h = int(region.get('h', 10) / 100 * height)
    return x, y, x + w, y + h


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
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200)
    if not images:
        return {'error': 'Failed to convert PDF'}, 500
    img_io = io.BytesIO()
    images[0].save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='plan.png')


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
        region = room.get('region_percent', {})
        if not region:
            continue
        x1, y1, x2, y2 = region_to_pixels(region, width, height)

        # Semi-transparent blue fill + border
        draw.rectangle([x1, y1, x2, y2],
                       fill=(59, 130, 246, 70),
                       outline=(59, 130, 246, 200),
                       width=2)

        # Room name label
        label = room.get('name', f'Помещение {i + 1}')
        draw.text((x1 + 6, y1 + 5), label, fill=(0, 0, 100, 230))

    result = Image.alpha_composite(img, overlay).convert('RGB')
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='annotated_rooms.png')


@app.route('/annotate-changes', methods=['POST'])
def annotate_changes():
    """
    Highlights changed rooms by classification on the floor plan.
    Input:  multipart/form-data  { image: <PNG binary>, rooms_json: <JSON string>, changes: <JSON string> }
    Output: PNG binary
    Colors: red = illegal, yellow = requires_approval, green = legal
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    for field in ('rooms_json', 'changes'):
        if field not in request.form:
            return {'error': f'{field} is required'}, 400

    img = Image.open(request.files['image']).convert('RGBA')
    rooms = json.loads(request.form['rooms_json'])
    changes = json.loads(request.form['changes'])
    width, height = img.size

    room_map = {r['id']: r for r in rooms if 'id' in r}

    fill_colors = {
        'illegal':           (239, 68,  68,  120),
        'requires_approval': (234, 179, 8,   120),
        'legal':             (34,  197, 94,  80),
    }
    border_colors = {
        'illegal':           (200, 30,  30,  230),
        'requires_approval': (180, 130, 0,   230),
        'legal':             (20,  150, 60,  180),
    }

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, change in enumerate(changes):
        room = room_map.get(change.get('room_id', ''), {})
        region = room.get('region_percent', {})
        if not region:
            continue

        cls = change.get('classification', 'legal')
        x1, y1, x2, y2 = region_to_pixels(region, width, height)

        draw.rectangle([x1, y1, x2, y2],
                       fill=fill_colors.get(cls, (128, 128, 128, 100)),
                       outline=border_colors.get(cls, (100, 100, 100, 200)),
                       width=2)

        # Numbered badge in top-left corner of room
        badge_x2, badge_y2 = x1 + 22, y1 + 22
        draw.rectangle([x1, y1, badge_x2, badge_y2],
                       fill=border_colors.get(cls, (100, 100, 100, 220)))
        draw.text((x1 + 5, y1 + 3), str(i + 1), fill=(255, 255, 255, 255))

    result = Image.alpha_composite(img, overlay).convert('RGB')
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', download_name='annotated_changes.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
