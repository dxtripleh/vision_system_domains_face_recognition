#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴인식 시스템 웹 인터페이스.

Flask 기반의 웹 UI를 제공하여 브라우저에서 얼굴인식 시스템을 사용할 수 있습니다.
"""

import os
import sys
import io
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from domains.face_recognition.core.services import FaceRecognitionService
from domains.face_recognition.core.entities import Person
from shared.vision_core.detection import FaceDetector
from shared.vision_core.recognition import FaceRecognizer
from common.logging import setup_logging
import logging

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vision_system_secret_key'
app.config['UPLOAD_FOLDER'] = 'data/temp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 전역 서비스 인스턴스
face_service = None
detector = None
recognizer = None

def initialize_services():
    """서비스 초기화"""
    global face_service, detector, recognizer
    
    try:
        face_service = FaceRecognitionService()
        detector = FaceDetector(detector_type="opencv")
        recognizer = FaceRecognizer(recognizer_type="arcface")
        
        logging.info("Web services initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        return False

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image: np.ndarray) -> str:
    """이미지를 Base64로 인코딩"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
    try:
        stats = face_service.get_statistics()
        persons = face_service.list_persons()
        
        return render_template('dashboard.html', 
                             stats=stats, 
                             persons=persons)
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/upload')
def upload_page():
    """이미지 업로드 페이지"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """이미지 업로드 및 처리"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # 파일 저장
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 이미지 로드
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # 얼굴 검출
        detections = detector.detect_faces(image)
        
        # 결과 준비
        results = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 얼굴 영역 추출
            face_region = detector.extract_face_region(image, bbox)
            
            # 얼굴 인식 시도
            recognition_result = None
            if recognizer:
                embedding = recognizer.extract_embedding(face_region)
                if embedding is not None:
                    # 기존 등록된 얼굴과 비교
                    persons = face_service.list_persons()
                    best_match = None
                    best_similarity = 0.0
                    
                    for person in persons:
                        faces = face_service.get_faces_by_person_id(person.person_id)
                        for face in faces:
                            similarity = recognizer.compute_similarity(
                                embedding, face.embedding.vector
                            )
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = {
                                    'person_id': person.person_id,
                                    'person_name': person.name,
                                    'similarity': similarity
                                }
                    
                    recognition_result = best_match
            
            # 얼굴 영역을 Base64로 인코딩
            face_base64 = encode_image_to_base64(face_region)
            
            results.append({
                'face_id': i,
                'bbox': bbox,
                'confidence': confidence,
                'face_image': face_base64,
                'recognition': recognition_result
            })
        
        # 원본 이미지를 Base64로 인코딩
        original_base64 = encode_image_to_base64(image)
        
        # 🔧 수정: 웹에서도 파일을 보존 (배치 처리 가능하도록)
        # os.remove(filepath)  # 주석 처리 - 파일 유지
        
        return jsonify({
            'success': True,
            'original_image': original_base64,
            'detections': results,
            'total_faces': len(results),
            'saved_file': filename  # 저장된 파일명 반환
        })
        
    except Exception as e:
        logging.error(f"Upload processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/register_person', methods=['POST'])
def register_person():
    """새 인물 등록"""
    try:
        data = request.get_json()
        
        name = data.get('name')
        metadata = data.get('metadata', {})
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        # 인물 등록
        person = face_service.register_person(name, metadata)
        
        return jsonify({
            'success': True,
            'person': {
                'person_id': person.person_id,
                'name': person.name,
                'metadata': person.metadata
            }
        })
        
    except Exception as e:
        logging.error(f"Person registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/register_face', methods=['POST'])
def register_face():
    """얼굴 등록"""
    try:
        data = request.get_json()
        
        person_id = data.get('person_id')
        image_data = data.get('image_data')
        
        if not person_id or not image_data:
            return jsonify({'error': 'Person ID and image data are required'}), 400
        
        # Base64 이미지 디코딩
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # 얼굴 등록
        success = face_service.register_face(person_id, image)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Face registration failed'}), 500
            
    except Exception as e:
        logging.error(f"Face registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/persons')
def list_persons():
    """등록된 인물 목록"""
    try:
        persons = face_service.list_persons()
        
        persons_data = []
        for person in persons:
            faces = face_service.get_faces_by_person_id(person.person_id)
            
            persons_data.append({
                'person_id': person.person_id,
                'name': person.name,
                'metadata': person.metadata,
                'face_count': len(faces)
            })
        
        return jsonify({
            'success': True,
            'persons': persons_data
        })
        
    except Exception as e:
        logging.error(f"List persons error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """시스템 통계"""
    try:
        stats = face_service.get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logging.error(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404 에러 핸들러"""
    return render_template('error.html', 
                         error="페이지를 찾을 수 없습니다."), 404

@app.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러"""
    return render_template('error.html', 
                         error="내부 서버 오류가 발생했습니다."), 500

def main():
    """메인 함수"""
    # 로깅 설정
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    # 필요한 디렉토리 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    os.makedirs('data/temp/face_staging', exist_ok=True)  # face_staging 폴더도 생성
    
    # 서비스 초기화
    if not initialize_services():
        logger.error("Failed to initialize services")
        return 1
    
    # 웹 서버 시작
    logger.info("Starting web interface...")
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 