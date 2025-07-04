#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Detection Model.

ONNX 기반 얼굴 검출 모델 클래스입니다.
RetinaFace 기반으로 얼굴을 검출하고 바운딩 박스와 랜드마크를 반환합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

try:
    import onnxruntime as ort
except ImportError:
    logging.error("ONNX Runtime이 설치되지 않았습니다. pip install onnxruntime을 실행하세요.")
    raise

logger = logging.getLogger(__name__)

class FaceDetectionModel:
    """얼굴 검출 모델 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """얼굴 검출 모델 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
            config: 검출 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        self.model_path = model_path or self._get_default_model_path()
        self.device_config = self._detect_hardware_environment()
        
        logger.info(f"하드웨어 환경: {self.device_config}")
        
        # ONNX 세션 초기화
        self.session = self._initialize_onnx_session()
        self.input_shape = self._get_input_shape()
        self.output_names = [output.name for output in self.session.get_outputs()] if self.session else []
        
        logger.info(f"얼굴 검출 모델 로딩 완료: {self.model_path}")
        logger.info(f"입력 형태: {self.input_shape}")
        logger.info(f"출력 이름: {self.output_names}")
        logger.info(f"더미 모델 모드: {self.session is None}")
    
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'input_size': (640, 640),
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'min_face_size': 80,
            'max_face_size': 640,
            'normalize': True
        }
    
    def _get_default_model_path(self) -> str:
        """기본 모델 경로 반환"""
        model_path = project_root / "models" / "weights" / "face_detection_retinaface_widerface_20250628.onnx"
        return str(model_path)
    
    def _detect_hardware_environment(self) -> Dict:
        """하드웨어 환경 감지"""
        import platform
        
        system = platform.system().lower()
        is_jetson = self._is_jetson_platform()
        gpu_available = False
        
        try:
            providers = ort.get_available_providers()
            gpu_available = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
        except Exception as e:
            logger.warning(f"GPU 정보 확인 실패: {e}")
        
        if is_jetson:
            return {
                'device': 'jetson',
                'providers': ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            }
        elif gpu_available:
            return {
                'device': 'gpu',
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
            }
        else:
            return {
                'device': 'cpu',
                'providers': ['CPUExecutionProvider']
            }
    
    def _is_jetson_platform(self) -> bool:
        """Jetson 플랫폼 감지"""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    def _initialize_onnx_session(self) -> ort.InferenceSession:
        """ONNX 세션 초기화"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        try:
            session = ort.InferenceSession(
                self.model_path,
                providers=self.device_config['providers']
            )
            return session
        except Exception as e:
            logger.warning(f"ONNX 모델 로딩 실패 (더미 모델일 가능성): {e}")
            # 더미 모델인 경우 None 반환
            return None
    
    def _get_input_shape(self) -> Tuple[int, int, int, int]:
        """입력 형태 반환"""
        if self.session is None:
            return (1, 3, 640, 640)  # 더미 모델용 기본값
        
        input_info = self.session.get_inputs()[0]
        shape = input_info.shape
        
        if shape[0] == -1:
            shape = (1,) + tuple(shape[1:])
        
        return tuple(shape)
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """이미지 전처리
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            (전처리된 텐서, 스케일 팩터 X, 스케일 팩터 Y)
        """
        original_height, original_width = image.shape[:2]
        input_size = self.config['input_size']
        
        # 리사이즈 및 패딩
        resized_image, scale_x, scale_y = self._resize_with_padding(image, input_size)
        
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 정규화
        if self.config['normalize']:
            normalized = rgb_image.astype(np.float32) / 255.0
        else:
            normalized = rgb_image.astype(np.float32)
        
        # 텐서 형태로 변환 (NCHW)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale_x, scale_y
    
    def _resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
        """종횡비를 유지하며 리사이즈 및 패딩"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # 스케일 팩터 계산
        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)
        
        # 새로운 크기 계산
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 리사이즈
        resized = cv2.resize(image, (new_width, new_height))
        
        # 패딩
        top = (target_height - new_height) // 2
        bottom = target_height - new_height - top
        left = (target_width - new_width) // 2
        right = target_width - new_width - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        return padded, scale_x, scale_y
    
    def postprocess(self, outputs: List[np.ndarray], scale_x: float, scale_y: float, 
                   original_shape: Tuple[int, int]) -> List[Dict]:
        """후처리: 검출 결과를 원본 이미지 좌표로 변환
        
        Args:
            outputs: ONNX 모델 출력
            scale_x: X축 스케일 팩터
            scale_y: Y축 스케일 팩터
            original_shape: 원본 이미지 크기 (height, width)
            
        Returns:
            검출된 얼굴 정보 리스트
        """
        detections = []
        
        # 더미 모델인 경우 가짜 검출 결과 반환
        if self.session is None:
            return self._create_dummy_detections(original_shape)
        
        # 실제 후처리 로직 (모델에 따라 달라질 수 있음)
        try:
            # 예시: RetinaFace 출력 형태에 따른 처리
            # 실제 모델의 출력 형태에 맞게 수정 필요
            if len(outputs) > 0:
                boxes = outputs[0]  # 바운딩 박스
                scores = outputs[1] if len(outputs) > 1 else None  # 신뢰도
                landmarks = outputs[2] if len(outputs) > 2 else None  # 랜드마크
                
                # NMS 적용 및 좌표 변환
                detections = self._apply_nms_and_transform(
                    boxes, scores, landmarks, scale_x, scale_y, original_shape
                )
        
        except Exception as e:
            logger.error(f"후처리 중 오류: {e}")
            detections = []
        
        return detections
    
    def _create_dummy_detections(self, original_shape: Tuple[int, int]) -> List[Dict]:
        """더미 검출 결과 생성 (개발용)"""
        height, width = original_shape
        
        # 중앙에 가짜 얼굴 검출 결과 생성
        center_x = width // 2
        center_y = height // 2
        face_size = min(width, height) // 4
        
        dummy_detection = {
            'bbox': [
                center_x - face_size // 2,  # x1
                center_y - face_size // 2,  # y1
                center_x + face_size // 2,  # x2
                center_y + face_size // 2   # y2
            ],
            'confidence': 0.95,
            'landmarks': [
                [center_x - 20, center_y - 10],  # 왼쪽 눈
                [center_x + 20, center_y - 10],  # 오른쪽 눈
                [center_x, center_y],            # 코
                [center_x - 15, center_y + 15],  # 왼쪽 입꼬리
                [center_x + 15, center_y + 15]   # 오른쪽 입꼬리
            ]
        }
        
        logger.info("더미 검출 결과 생성됨 (개발용)")
        return [dummy_detection]
    
    def _apply_nms_and_transform(self, boxes: np.ndarray, scores: np.ndarray, 
                                landmarks: np.ndarray, scale_x: float, scale_y: float,
                                original_shape: Tuple[int, int]) -> List[Dict]:
        """NMS 적용 및 좌표 변환"""
        detections = []
        
        # 실제 NMS 로직 구현 필요
        # 여기서는 간단한 예시만 제공
        
        return detections
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """얼굴 검출 수행
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            검출된 얼굴 정보 리스트
            각 얼굴: {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'landmarks': [[x, y], ...] (5개 점)
            }
        """
        if image is None or image.size == 0:
            logger.warning("유효하지 않은 입력 이미지")
            return []
        
        original_shape = image.shape[:2]
        
        # 전처리
        input_tensor, scale_x, scale_y = self.preprocess(image)
        
        # 추론
        if self.session is None:
            # 더미 모델인 경우
            outputs = []
        else:
            try:
                outputs = self.session.run(
                    self.output_names, 
                    {self.session.get_inputs()[0].name: input_tensor}
                )
            except Exception as e:
                logger.error(f"추론 중 오류: {e}")
                outputs = []
        
        # 후처리
        detections = self.postprocess(outputs, scale_x, scale_y, original_shape)
        
        # 필터링
        filtered_detections = self._filter_detections(detections)
        
        logger.info(f"얼굴 검출 완료: {len(filtered_detections)}개 검출")
        return filtered_detections
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """검출 결과 필터링"""
        filtered = []
        
        for detection in detections:
            # 신뢰도 필터링
            if detection['confidence'] < self.config['confidence_threshold']:
                continue
            
            # 얼굴 크기 필터링
            bbox = detection['bbox']
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_size = max(face_width, face_height)
            
            if face_size < self.config['min_face_size'] or face_size > self.config['max_face_size']:
                continue
            
            filtered.append(detection)
        
        return filtered
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'output_names': self.output_names,
            'device_config': self.device_config,
            'config': self.config,
            'is_dummy': self.session is None
        } 