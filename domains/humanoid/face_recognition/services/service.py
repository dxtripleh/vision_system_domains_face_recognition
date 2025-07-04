#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Service.

얼굴 검출과 인식을 통합하는 서비스 클래스입니다.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from ..models.face_detection_model import FaceDetectionModel
from ..models.face_recognition_model import FaceRecognitionModel

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """통합 얼굴인식 서비스"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.detector = FaceDetectionModel(config=self.config.get('detection', {}))
        self.recognizer = FaceRecognitionModel(config=self.config.get('recognition', {}))
        self.stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'average_processing_time': 0.0,
            'last_processing_time': 0.0
        }
        logger.info("얼굴인식 서비스 초기화 완료")
    
    def _get_default_config(self) -> Dict:
        return {
            'detection': {'confidence_threshold': 0.5, 'min_face_size': 80},
            'recognition': {'similarity_threshold': 0.6, 'enable_quality_check': True},
            'visualization': {'draw_bbox': True, 'bbox_color': (0, 255, 0)},
            'performance': {'enable_profiling': True, 'log_every_n_frames': 30}
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return self._create_empty_result(frame)
        
        try:
            detections = self.detector.detect_faces(frame)
            faces = []
            
            for detection in detections:
                face_info = self._process_single_face(frame, detection)
                if face_info:
                    faces.append(face_info)
            
            frame_with_results = self._visualize_results(frame.copy(), faces)
            processing_time = time.time() - start_time
            self._update_stats(len(faces), processing_time)
            
            return {
                'faces': faces,
                'processing_time': processing_time,
                'frame_with_results': frame_with_results,
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}")
            return self._create_empty_result(frame)
    
    def _process_single_face(self, frame: np.ndarray, detection: Dict) -> Optional[Dict]:
        try:
            face_image = self._extract_face_region(frame, detection)
            if face_image is None:
                return None
            
            embedding = self.recognizer.infer(face_image)
            identity = self._match_identity(embedding)
            
            return {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'landmarks': detection.get('landmarks', []),
                'embedding': embedding,
                'identity': identity
            }
        except Exception as e:
            logger.error(f"얼굴 처리 중 오류: {e}")
            return None
    
    def _extract_face_region(self, frame: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            face_image = frame[y1:y2, x1:x2]
            return face_image if face_image.size > 0 else None
        except Exception as e:
            logger.error(f"얼굴 영역 추출 중 오류: {e}")
            return None
    
    def _match_identity(self, embedding: np.ndarray) -> Dict:
        return {
            'person_id': 'unknown',
            'person_name': '미확인',
            'similarity': 0.0,
            'is_known': False
        }
    
    def _visualize_results(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        viz_config = self.config['visualization']
        
        for face in faces:
            bbox = face['bbox']
            confidence = face['confidence']
            identity = face.get('identity', {})
            
            if viz_config['draw_bbox']:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), viz_config['bbox_color'], 2)
                
                text = f"{identity.get('person_name', '미확인')} ({confidence:.2f})"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self._draw_stats_overlay(frame)
        return frame
    
    def _draw_stats_overlay(self, frame: np.ndarray):
        stats_text = [
            f"Frames: {self.stats['total_frames']}",
            f"Faces: {self.stats['total_faces_detected']}",
            f"FPS: {1.0/self.stats['average_processing_time']:.1f}" if self.stats['average_processing_time'] > 0 else "FPS: 0.0"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + (i * 25)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _update_stats(self, num_faces: int, processing_time: float):
        self.stats['total_frames'] += 1
        self.stats['total_faces_detected'] += num_faces
        self.stats['last_processing_time'] = processing_time
        
        alpha = 0.1
        if self.stats['average_processing_time'] == 0:
            self.stats['average_processing_time'] = processing_time
        else:
            self.stats['average_processing_time'] = (
                alpha * processing_time + (1 - alpha) * self.stats['average_processing_time']
            )
    
    def _create_empty_result(self, frame: Optional[np.ndarray]) -> Dict:
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return {
            'faces': [],
            'processing_time': 0.0,
            'frame_with_results': frame,
            'stats': self.stats.copy()
        }
    
    def get_service_info(self) -> Dict:
        return {
            'detector_info': self.detector.get_model_info(),
            'recognizer_info': self.recognizer.get_model_info(),
            'config': self.config,
            'stats': self.stats.copy()
        } 