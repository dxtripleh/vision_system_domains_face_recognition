#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Detection Service.

얼굴 검출을 위한 도메인 서비스입니다.
"""

# 표준 라이브러리
import os
import time
import uuid
from typing import List, Optional, Dict, Any

# 서드파티 라이브러리
import cv2
import numpy as np

# 프로젝트 모듈
from common.logging import get_logger
from ..entities.face import Face
from ..entities.face_detection_result import FaceDetectionResult
from ..value_objects.bounding_box import BoundingBox
from ..value_objects.confidence_score import ConfidenceScore

logger = get_logger(__name__)


class FaceDetectionService:
    """얼굴 검출 서비스"""
    
    def __init__(self, detector=None, config: Optional[Dict[str, Any]] = None, use_mock: bool = False):
        """
        서비스 초기화
        
        Args:
            detector: 얼굴 검출기 (의존성 주입)
            config: 설정 딕셔너리
            use_mock: Mock 구현 사용 여부 (테스트용)
        """
        self.config = config or {}
        self.use_mock = use_mock
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.min_face_size = self.config.get('min_face_size', (80, 80))
        self.max_faces = self.config.get('max_faces', 10)
        
        # 검출기 초기화
        if detector is not None:
            self.detector = detector
        else:
            self.detector = self._initialize_default_detector()
        
        logger.info(f"FaceDetectionService initialized with config: {self.config}")
    
    def _initialize_default_detector(self):
        """기본 검출기 초기화 (RetinaFace 우선, 실패시 OpenCV)"""
        try:
            # RetinaFace 시도
            from ...infrastructure.models.retinaface_detector import RetinaFaceDetector
            
            model_path = self.config.get('detection_model_path', 'models/weights/face_detection_retinaface_mnet025_20250628.onnx')
            if os.path.exists(model_path):
                detector = RetinaFaceDetector(
                    model_path=model_path,
                    confidence_threshold=self.min_confidence,
                    use_gpu=self.config.get('use_gpu', True)
                )
                
                if detector.load_model():
                    logger.info("RetinaFace detector loaded successfully")
                    return detector
                else:
                    logger.warning("Failed to load RetinaFace, falling back to OpenCV")
            else:
                logger.warning(f"RetinaFace model not found: {model_path}, falling back to OpenCV")
                
        except Exception as e:
            logger.warning(f"Error initializing RetinaFace: {str(e)}, falling back to OpenCV")
        
        # OpenCV Haar Cascade 폴백
        logger.info("Using OpenCV Haar Cascade as fallback")
        return None
    
    def detect_faces(self, image: np.ndarray, image_id: Optional[str] = None) -> FaceDetectionResult:
        """
        이미지에서 얼굴을 검출합니다.
        
        Args:
            image: 입력 이미지 (BGR 형식)
            image_id: 이미지 식별자 (선택사항)
            
        Returns:
            FaceDetectionResult: 검출 결과
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
        """
        if image is None or image.size == 0:
            raise ValueError("유효하지 않은 이미지입니다")
        
        start_time = time.time()
        
        if image_id is None:
            image_id = str(uuid.uuid4())
        
        logger.debug(f"얼굴 검출 시작 - 이미지 ID: {image_id}")
        
        try:
            # 얼굴 검출 (RetinaFace 우선, 실패시 OpenCV)
            if self.detector is not None:
                # RetinaFace 사용
                faces = self._detect_faces_retinaface(image)
                model_name = "retinaface"
            else:
                # OpenCV Haar Cascade 사용
                processed_image = self._preprocess_image(image)
                faces = self._detect_faces_opencv(processed_image)
                model_name = "opencv_haarcascade"
            
            # 후처리 및 필터링
            filtered_faces = self._postprocess_faces(faces, image.shape)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # 결과 생성
            result = FaceDetectionResult(
                image_id=image_id,
                faces=filtered_faces,
                processing_time_ms=processing_time_ms,
                model_name=model_name,
                image_metadata={
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "channels": image.shape[2] if len(image.shape) > 2 else 1
                }
            )
            
            logger.info(f"얼굴 검출 완료 - {len(filtered_faces)}개 검출, {processing_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"얼굴 검출 중 오류 발생: {str(e)}")
            raise
    
    def detect_faces_batch(self, images: List[np.ndarray], 
                          image_ids: Optional[List[str]] = None) -> List[FaceDetectionResult]:
        """
        여러 이미지에서 배치로 얼굴을 검출합니다.
        
        Args:
            images: 입력 이미지 리스트
            image_ids: 이미지 식별자 리스트 (선택사항)
            
        Returns:
            List[FaceDetectionResult]: 검출 결과 리스트
        """
        if not images:
            return []
        
        if image_ids is None:
            image_ids = [str(uuid.uuid4()) for _ in images]
        
        if len(images) != len(image_ids):
            raise ValueError("이미지 개수와 ID 개수가 일치하지 않습니다")
        
        results = []
        for image, image_id in zip(images, image_ids):
            try:
                result = self.detect_faces(image, image_id)
                results.append(result)
            except Exception as e:
                logger.error(f"배치 처리 중 오류 (이미지 ID: {image_id}): {str(e)}")
                # 오류가 발생한 경우 빈 결과 추가
                results.append(FaceDetectionResult(
                    image_id=image_id,
                    faces=[],
                    processing_time_ms=0,
                    model_name="unknown",
                    image_metadata={"error": str(e)}
                ))
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 히스토그램 평활화
        gray = cv2.equalizeHist(gray)
        
        return gray
    
    def _detect_faces_retinaface(self, image: np.ndarray) -> List[Dict]:
        """RetinaFace를 사용한 얼굴 검출"""
        try:
            if self.detector is None:
                logger.error("RetinaFace detector not initialized")
                return []
            
            # RetinaFace 검출
            detections = self.detector.detect(image)
            
            # 결과 형식 변환
            detected_faces = []
            for detection in detections:
                bbox = detection.get('bbox', [])
                confidence = detection.get('confidence', 0.0)
                landmarks = detection.get('landmarks', [])
                
                if len(bbox) == 4:  # x, y, w, h
                    detected_faces.append({
                        'bbox': tuple(bbox),
                        'confidence': confidence,
                        'landmarks': landmarks
                    })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"RetinaFace 얼굴 검출 중 오류: {str(e)}")
            return []
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict]:
        """OpenCV Haar Cascade를 사용한 얼굴 검출 (폴백)"""
        try:
            # Haar Cascade 분류기 로드
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 얼굴 검출
            faces = face_cascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 결과 형식 변환
            detected_faces = []
            for (x, y, w, h) in faces:
                detected_faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # Haar Cascade는 신뢰도를 제공하지 않으므로 고정값
                    'landmarks': []  # Haar Cascade는 랜드마크를 제공하지 않음
                })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"OpenCV 얼굴 검출 중 오류: {str(e)}")
            return []
    
    def _postprocess_faces(self, raw_faces: List[Dict], image_shape: tuple) -> List[Face]:
        """검출된 얼굴들을 후처리하고 Face 엔티티로 변환"""
        faces = []
        
        for i, face_data in enumerate(raw_faces):
            try:
                # 바운딩 박스 생성
                x, y, w, h = face_data['bbox']
                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                
                # 신뢰도 생성
                confidence = ConfidenceScore(value=face_data['confidence'])
                
                # 최소 신뢰도 확인
                if confidence.value < self.min_confidence:
                    continue
                
                # 바운딩 박스가 이미지 범위 내에 있는지 확인
                if not self._is_valid_bbox(bbox, image_shape):
                    continue
                
                # Face 엔티티 생성
                face = Face(
                    face_id=str(uuid.uuid4()),
                    person_id=None,
                    embedding=None,  # 나중에 인식 단계에서 추가
                    confidence=confidence.value,
                    bbox=bbox,
                    landmarks=face_data.get('landmarks', []),
                    quality_score=self._calculate_quality_score(bbox, image_shape),
                    created_at=time.time()
                )
                
                faces.append(face)
                
            except Exception as e:
                logger.warning(f"얼굴 {i} 후처리 중 오류: {str(e)}")
                continue
        
        # 신뢰도 순으로 정렬하고 최대 개수 제한
        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces[:self.max_faces]
    
    def _is_valid_bbox(self, bbox: BoundingBox, image_shape: tuple) -> bool:
        """바운딩 박스가 이미지 범위 내에 있는지 확인"""
        height, width = image_shape[:2]
        
        return (0 <= bbox.x < width and 
                0 <= bbox.y < height and
                bbox.x + bbox.width <= width and
                bbox.y + bbox.height <= height and
                bbox.width >= self.min_face_size[0] and
                bbox.height >= self.min_face_size[1])
    
    def _calculate_quality_score(self, bbox: BoundingBox, image_shape: tuple) -> float:
        """얼굴 품질 점수 계산 (간단한 버전)"""
        # 얼굴 크기 기반 품질 점수
        face_area = bbox.area
        image_area = image_shape[0] * image_shape[1]
        
        # 얼굴이 이미지에서 차지하는 비율
        area_ratio = face_area / image_area
        
        # 0.01 ~ 0.25 비율을 0.5 ~ 1.0 점수로 매핑
        if area_ratio < 0.01:
            return 0.3
        elif area_ratio > 0.25:
            return 1.0
        else:
            return 0.5 + (area_ratio - 0.01) / (0.25 - 0.01) * 0.5
    
    def validate_face_quality(self, face: Face) -> bool:
        """얼굴 품질 검증"""
        min_quality = self.config.get('min_quality_score', 0.5)
        return face.quality_score >= min_quality
    
    def get_statistics(self) -> Dict[str, Any]:
        """서비스 통계 정보 반환"""
        return {
            "model_name": "opencv_haarcascade",
            "min_confidence": self.min_confidence,
            "min_face_size": self.min_face_size,
            "max_faces": self.max_faces,
            "config": self.config
        } 