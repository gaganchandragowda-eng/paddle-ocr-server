from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import cv2
import numpy as np
import base64
import re
import io
from PIL import Image

app = Flask(__name__)
CORS(app)
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    return thresh

def clean_number(text):
    try:
        return float(text.replace(',','').replace('₹','').replace('%','').strip())
    except:
        return 0

def is_number(text):
    try:
        float(text.replace(',','').replace('₹','').replace('%','').strip())
        return True
    except:
        return False

def parse_rows(ocr_result):
    if not ocr_result or not ocr_result[0]:
        return []
    lines = []
    for line in ocr_result[0]:
        bbox, (text, conf) = line[0], line[1]
        y = (bbox[0][1] + bbox[2][1]) / 2
        x = (bbox[0][0] + bbox[2][0]) / 2
        lines.append({'text': text, 'y': y, 'x': x})
    lines.sort(key=lambda l: l['y'])
    rows, cur, last_y = [], [], -1
    for item in lines:
        if last_y == -1 or abs(item['y'] - last_y) <= 15:
            cur.append(item)
        else:
            if cur:
                rows.append(sorted(cur, key=lambda l: l['x']))
            cur = [item]
        last_y = item['y']
    if cur:
        rows.append(sorted(cur, key=lambda l: l['x']))
    return rows

def extract(rows):
    full_text = ' '.join(i['text'] for r in rows for i in r)

    gstins = re.findall(r'\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b', full_text)
    bill_match = re.search(r'(?:invoice|bill|inv)[\s#.:no]*([A-Z0-9\-\/]+)', full_text, re.I)
    date_match = re.search(r'(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})', full_text)

    bill_date = ''
    if date_match:
        p = date_match.group(1).split('/' if '/' in date_match.group(1) else '-')
        if len(p) == 3:
            y = '20'+p[2] if len(p[2])==2 else p[2]
            bill_date = f"{y}-{p[1].zfill(2)}-{p[0].zfill(2)}"

    supplier_name = rows[0][0]['text'] if rows else ''
    tax_type = 'IGST' if 'IGST' in full_text else 'CGST_SGST' if 'CGST' in full_text else 'UNKNOWN'

    # Find item rows - rows with numbers that look like qty + rate
    items = []
    header_keywords = {'description','particular','item','hsn','qty','rate','amount','total','gst','sl','no'}

    for row in rows:
        texts = [i['text'] for i in row]
        row_text = ' '.join(texts).lower()

        if any(k in row_text for k in header_keywords):
            continue

        numbers = [t for t in texts if is_number(t)]
        words = [t for t in texts if not is_number(t) and len(t) > 2]

        if len(numbers) >= 2 and words:
            description = ' '.join(words)
            nums = [clean_number(n) for n in numbers]
            nums_sorted = sorted(nums)

            qty = nums[0] if nums[0] < 10000 else 1
            rate = nums[1] if len(nums) > 1 else 0

            gst_match = re.search(r'(\d+(?:\.\d+)?)\s*%', ' '.join(texts))
            gst_rate = float(gst_match.group(1)) if gst_match else 18

            disc_match = re.search(r'disc[ount]*[\s:]*(\d+(?:\.\d+)?)', row_text)
            discount = float(disc_match.group(1)) if disc_match else 0

            discounted_rate = rate * (1 - discount/100)
            taxable = round(discounted_rate * qty, 2)
            gst_amount = round(taxable * gst_rate / 100, 2)

            items.append({
                'description': description,
                'hsn_code': '',
                'quantity': qty,
                'unit': 'Nos',
                'rate': rate,
                'discount': discount,
                'gst_rate': gst_rate,
                'taxable_amount': taxable,
                'gst_amount': gst_amount,
                'total_amount': round(taxable + gst_amount, 2)
            })

    taxable_total = round(sum(i['taxable_amount'] for i in items), 2)
    gst_total = round(sum(i['gst_amount'] for i in items), 2)

    return {
        'supplier_name': supplier_name,
        'supplier_gstin': gstins[0] if gstins else '',
        'buyer_gstin': gstins[1] if len(gstins) > 1 else '',
        'bill_number': bill_match.group(1) if bill_match else '',
        'bill_date': bill_date,
        'tax_type': tax_type,
        'items': items,
        'taxable_total': taxable_total,
        'gst_total': gst_total,
        'grand_total': round(taxable_total + gst_total, 2)
    }

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        img_data = data.get('image', '')
        img_data = re.sub(r'^data:image\/\w+;base64,', '', img_data)
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed = preprocess(img_cv)
        result = ocr.ocr(processed, cls=True)
        rows = parse_rows(result)
        invoice_data = extract(rows)
        return jsonify(invoice_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
