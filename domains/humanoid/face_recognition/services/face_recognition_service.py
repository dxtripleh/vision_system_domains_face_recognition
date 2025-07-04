#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Service.

얼굴 검출과 인식을 통합하는 서비스 클래스입니다.
전체 얼굴인식 파이프라인을 관리하고 실행합니다.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from ..models.face_detection_model import FaceDetectionModel
from ..models.face_recognition_model import FaceRecognitionModel

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """통합 얼굴인식 서비스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """서비스 초기화
        
        Args:
            config: 서비스 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        
        # 모델 초기화
        self.detector = FaceDetectionModel(
            config=self.config.get('detection', {})
        )
        self.recognizer = FaceRecognitionModel(
            config=self.config.get('recognition', {})
        )
        
        # 얼굴 데이터베이스 (향후 구현)
        self.face_database = None
        
        # 성능 통계
        self.stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'average_processing_time': 0.0,
            'last_processing_time': 0.0
        }
        
        logger.info("얼굴인식 서비스 초기화 완료")
        logger.info(f"검출 모델: {self.detector.get_model_info()['is_dummy']}")
        logger.info(f"인식 모델: {self.recognizer.get_model_info()}")
    
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'detection': {
                'confidence_threshold': 0.5,
                'min_face_size': 80,
                'max_face_size': 640
            },
            'recognition': {
                'similarity_threshold': 0.6,
                'enable_quality_check': True,
                'quality_threshold': 0.5
            },
            'visualization': {
                'draw_bbox': True,
                'draw_landmarks': True,
                'draw_confidence': True,
                'draw_identity': True,
                'bbox_color': (0, 255, 0),
                'landmark_color': (255, 0, 0),
                'text_color': (255, 255, 255)
            },
            'performance': {
                'enable_profiling': True,
                'log_every_n_frames': 30
            }
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """프레임 처리: 검출 → 인식 → 결과 반환
        
        Args:
            frame: 입력 프레임 (BGR 형식)
            
        Returns:
            처리 결과 딕셔너리: {
                'faces': List[Dict],  # 검출된 얼굴들
                'processing_time': float,
                'frame_with_results': np.ndarray
            }
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            logger.warning("유효하지 않은 프레임")
            return self._create_empty_result(frame)
        
        try:
            # 1. 얼굴 검출
            detections = self.detector.detect_faces(frame)
            
            # 2. 각 얼굴에 대해 인식 수행
            faces = []
            for detection in detections:
                face_info = self._process_single_face(frame, detection)
                if face_info:
                    faces.append(face_info)
            
            # 3. 결과 시각화
            frame_with_results = self._visualize_results(frame.copy(), faces)
            
            # 4. 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_stats(len(faces), processing_time)
            
            result = {
                'faces': faces,
                'processing_time': processing_time,
                'frame_with_results': frame_with_results,
                'stats': self.stats.copy()
            }
            
            # 주기적 로깅
            if self.config['performance']['enable_profiling']:
                self._log_performance_if_needed()
            
            return result
            
        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}")
            return self._create_empty_result(frame)
    
    def _process_single_face(self, frame: np.ndarray, detection: Dict) -> Optional[Dict]:
        """단일 얼굴 처리
        
        Args:
            frame: 원본 프레임
            detection: 검출 결과
            
        Returns:
            얼굴 정보 딕셔너리 또는 None
        """
        try:
            # 얼굴 영역 추출
            face_image = self._extract_face_region(frame, detection)
            if face_image is None:
                return None
            
            # 얼굴 품질 평가 (선택적)
            if self.config['recognition']['enable_quality_check']:
                quality_score = self._assess_face_quality(face_image)
                if quality_score < self.config['recognition']['quality_threshold']:
                    logger.debug(f"얼굴 품질 부족: {quality_score:.2f}")
                    return None
            else:
                quality_score = 1.0
            
            # 얼굴 임베딩 생성
            embedding = self.recognizer.infer(face_image)
            
            # 데이터베이스 매칭 (향후 구현)
            identity = self._match_identity(embedding)
            
            face_info = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'landmarks': detection.get('landmarks', []),
                'embedding': embedding,
                'identity': identity,
                'quality_score': quality_score
            }
            
            return face_info
            
        except Exception as e:
            logger.error(f"얼굴 처리 중 오류: {e}")
            return None
    
    def _extract_face_region(self, frame: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """얼굴 영역 추출"""
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # 바운딩 박스 유효성 검사
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # 얼굴 영역 추출
            face_image = frame[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return None
            
            return face_image
            
        except Exception as e:
            logger.error(f"얼굴 영역 추출 중 오류: {e}")
            return None
    
    def _assess_face_quality(self, face_image: np.ndarray) -> float:
        """얼굴 품질 평가
        
        Args:
            face_image: 얼굴 이미지
            
        Returns:
            품질 점수 (0.0 ~ 1.0)
        """
        try:
            # 간단한 품질 평가 (블러, 밝기 등)
            
            # 1. 블러 검사 (라플라시안 분산)
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_normalized = min(blur_score / 1000.0, 1.0)  # 정규화
            
            # 2. 밝기 검사
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # 128 기준
            
            # 3. 크기 검사
            height, width = face_image.shape[:2]
            size_score = min(min(height, width) / 80.0, 1.0)  # 80px 기준
            
            # 종합 점수
            quality_score = (blur_normalized + brightness_score + size_score) / 3.0
            
            return max(0.0, min(quality_score, 1.0))
            
        except Exception as e:
            logger.error(f"품질 평가 중 오류: {e}")
            return 0.5  # 기본값
    
    def _match_identity(self, embedding: np.ndarray) -> Dict:
        """임베딩을 이용한 신원 매칭
        
        Args:
            embedding: 얼굴 임베딩 벡터
            
        Returns:
            신원 정보 딕셔너리
        """
        # 현재는 더미 구현 (향후 FaceDatabase 클래스와 연동)
        return {
            'person_id': 'unknown',
            'person_name': '미확인',
            'similarity': 0.0,
            'is_known': False
        }
    
    def _visualize_results(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """결과 시각화
        
        Args:
            frame: 원본 프레임
            faces: 얼굴 정보 리스트
            
        Returns:
            시각화된 프레임
        """
        viz_config = self.config['visualization']
        
        for face in faces:
            bbox = face['bbox']
            confidence = face['confidence']
            landmarks = face.get('landmarks', [])
            identity = face.get('identity', {})
            quality_score = face.get('quality_score', 0.0)
            
            # 바운딩 박스 그리기
            if viz_config['draw_bbox']:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), viz_config['bbox_color'], 2)
            
            # 랜드마크 그리기
            if viz_config['draw_landmarks'] and landmarks:
                for landmark in landmarks:
                    x, y = landmark
                    cv2.circle(frame, (int(x), int(y)), 2, viz_config['landmark_color'], -1)
            
            # 텍스트 정보 그리기
            if viz_config['draw_confidence'] or viz_config['draw_identity']:
                text_lines = []
                
                if viz_config['draw_confidence']:
                    text_lines.append(f"Conf: {confidence:.2f}")
                    text_lines.append(f"Qual: {quality_score:.2f}")
                
                if viz_config['draw_identity']:
                    person_name = identity.get('person_name', '미확인')
                    similarity = identity.get('similarity', 0.0)
                    text_lines.append(f"{person_name}")
                    if similarity > 0:
                        text_lines.append(f"Sim: {similarity:.2f}")
                
                # 텍스트 그리기
                x1, y1 = bbox[:2]
                for i, text in enumerate(text_lines):
                    y_offset = y1 - 10 - (i * 20)
                    if y_offset > 0:
                        cv2.putText(frame, text, (x1, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  viz_config['text_color'], 1)
        
        # 전체 통계 정보 그리기
        self._draw_stats_overlay(frame)
        
        return frame
    
    def _draw_stats_overlay(self, frame: np.ndarray):
        """통계 정보 오버레이"""
        height, width = frame.shape[:2]
        
        stats_text = [
            f"Frames: {self.stats['total_frames']}",
            f"Faces: {self.stats['total_faces_detected']}",
            f"FPS: {1.0/self.stats['average_processing_time']:.1f}" if self.stats['average_processing_time'] > 0 else "FPS: 0.0",
            f"Time: {self.stats['last_processing_time']*1000:.1f}ms"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + (i * 25)
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 255), 2)
    
    def _update_stats(self, num_faces: int, processing_time: float):
        """성능 통계 업데이트"""
        self.stats['total_frames'] += 1
        self.stats['total_faces_detected'] += num_faces
        self.stats['last_processing_time'] = processing_time
        
        # 이동 평균으로 평균 처리 시간 계산
        alpha = 0.1  # 가중치
        if self.stats['average_processing_time'] == 0:
            self.stats['average_processing_time'] = processing_time
        else:
            self.stats['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['average_processing_time']
            )
    
    def _log_performance_if_needed(self):
        """주기적 성능 로깅"""
        log_interval = self.config['performance']['log_every_n_frames']
        
        if self.stats['total_frames'] % log_interval == 0:
            fps = 1.0 / self.stats['average_processing_time'] if self.stats['average_processing_time'] > 0 else 0
            logger.info(
                f"성능 통계 - "
                f"프레임: {self.stats['total_frames']}, "
                f"검출된 얼굴: {self.stats['total_faces_detected']}, "
                f"평균 FPS: {fps:.1f}, "
                f"평균 처리시간: {self.stats['average_processing_time']*1000:.1f}ms"
            )
    
    def _create_empty_result(self, frame: Optional[np.ndarray]) -> Dict:
        """빈 결과 생성"""
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return {
            'faces': [],
            'processing_time': 0.0,
            'frame_with_results': frame,
            'stats': self.stats.copy()
        }
    
    def get_service_info(self) -> Dict:
        """서비스 정보 반환"""
        return {
            'detector_info': self.detector.get_model_info(),
            'recognizer_info': self.recognizer.get_model_info(),
            'config': self.config,
            'stats': self.stats.copy()
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'average_processing_time': 0.0,
            'last_processing_time': 0.0
        }
        logger.info("성능 통계 초기화됨") 