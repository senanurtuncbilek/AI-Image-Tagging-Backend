from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.detector import YOLODetector
from utils.keyword_gen import generate_keywords
from config import Config
import os
import time
import logging
from collections import Counter
from werkzeug.utils import secure_filename

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

VALID_TOKEN = "1234567890987654321"

detector = None

def init_detector():
    global detector
    try:
        logger.info("YOLO detector başlatılıyor...")
        detector = YOLODetector()
        logger.info("YOLO detector hazır!")
    except Exception as e:
        logger.error(f"Detector başlatma hatası: {str(e)}")
        raise

@app.before_request
def before_request():
    global detector
    if detector is None:
        init_detector()

@app.route('/process', methods=['POST'])
def process_image():
    # --- GÜVENLİK KONTROLÜ (BEARER TOKEN) ---
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Yetkisiz erişim denemesi: Token bulunamadı.")
        return jsonify({'success': False, 'error': 'Bearer token gerekli!'}), 401
    
    token = auth_header.split(" ")[1]
    if token != VALID_TOKEN:
        logger.warning(f"Geçersiz token denemesi: {token}")
        return jsonify({'success': False, 'error': 'Geçersiz yetki!'}), 401
    # ---------------------------------------

    start_time = time.time()
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Resim dosyası yüklenmedi'}), 400
        
        file = request.files['image']
        temp_dir = 'temp_uploads'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        results = detector.detect(temp_path, confidence=Config.CONFIDENCE_THRESHOLD)
        
        all_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    all_detections.append({
                        'class': result.names[class_id],
                        'confidence': float(box.conf[0])
                    })
        
        class_counter = Counter([det['class'] for det in all_detections])
        object_counts = dict(class_counter)
        avg_confidence = sum([det['confidence'] for det in all_detections]) / len(all_detections) if all_detections else 0
        keywords = generate_keywords(object_counts)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': True,
            'object_counts': object_counts,
            'keywords': keywords,
            'total_objects': len(all_detections),
            'confidence': round(avg_confidence, 2),
            'processing_time': round(time.time() - start_time, 2),
            'model_version': 'YOLOv8n'
        })
        
    except Exception as e:
        logger.error(f"İşleme hatası: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    init_detector()
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.DEBUG)