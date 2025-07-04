#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Factory Defect Detection Model.

YOLOv8 기반 ONNX 모델을 사용한 공장 불량 검출 모델입니다.
크로스 플랫폼 호환성을 고려하여 설계되었습니다.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

try:
    import onnxruntime as ort
except ImportError:
    logging.error("ONNX Runtime이 설치되지 않았습니다. pip install onnxruntime를 실행하세요.")
    raise

# 불량 유형 정의 (순환 import 방지)
DEFECT_TYPES = {
    'scratch': {'id': 0, 'name': '스크래치', 'color': [0, 0, 255]},
    'dent': {'id': 1, 'name': '함몰', 'color': [0, 255, 0]},
    'crack': {'id': 2, 'name': '균열', 'color': [255, 0, 0]},
    'discoloration': {'id': 3, 'name': '변색', 'color': [255, 255, 0]},
    'contamination': {'id': 4, 'name': '오염', 'color': [255, 0, 255]}
}

logger = logging.getLogger(__name__)

class DefectDetectionModel:
    """공장 불량 검출 모델 클래스."""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        모델 초기화.
        
        Args:
            model_path: ONNX 모델 파일 경로
            config: 모델 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        self.model_path = model_path or self._get_default_model_path()
        
        # 하드웨어 환경 감지 및 최적화
        self.device_config = self._detect_hardware_environment()
        logger.info(f"하드웨어 환경: {self.device_config}")
        
        # ONNX 세션 초기화
        self.session = self._initialize_onnx_session()
        
        # 모델 정보
        self.input_shape = self._get_input_shape()
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"모델 로딩 완료: {self.model_path}")
        logger.info(f"입력 형태: {self.input_shape}")
        logger.info(f"출력 이름: {self.output_names}")
    
    def _get_default_config(self) -> Dict:
        """기본 설정을 반환합니다."""
        return {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'input_size': (640, 640),
            'max_detections': 100
        }
    
    def _get_default_model_path(self) -> str:
        """기본 모델 경로를 반환합니다."""
        model_path = project_root / "models" / "weights" / "defect_detection_yolov8n_factory.onnx"
        return str(model_path)
    
    def _detect_hardware_environment(self) -> Dict:
        """하드웨어 환경을 감지하고 최적화 설정을 반환합니다."""
        import platform
        import psutil
        
        system = platform.system().lower()
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total // (1024**3)
        
        # Jetson 환경 감지
        is_jetson = self._is_jetson_platform()
        
        # GPU 사용 가능 여부 확인
        gpu_available = False
        gpu_memory = 0
        
        try:
            # ONNX Runtime GPU 프로바이더 확인
            providers = ort.get_available_providers()
            gpu_available = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
            
            if gpu_available:
                # GPU 메모리 정보 (대략적 추정)
                if 'CUDAExecutionProvider' in providers:
                    gpu_memory = 8  # 기본값
        except Exception as e:
            logger.warning(f"GPU 정보 확인 실패: {e}")
        
        # 최적화 설정 결정
        if is_jetson:
            return {
                'device': 'jetson',
                'providers': ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
                'optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                'batch_size': 1
            }
        elif gpu_available and gpu_memory >= 8:
            return {
                'device': 'gpu',
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                'optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                'batch_size': 4
            }
        else:
            return {
                'device': 'cpu',
                'providers': ['CPUExecutionProvider'],
                'optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                'batch_size': 1
            }
    
    def _is_jetson_platform(self) -> bool:
        """Jetson 플랫폼인지 확인합니다."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    def _initialize_onnx_session(self) -> ort.InferenceSession:
        """ONNX 세션을 초기화합니다."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 세션 옵션 설정
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = self.device_config['optimization_level']
        session_options.log_severity_level = 2  # WARNING 레벨
        
        # 세션 생성
        session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=self.device_config['providers']
        )
        
        return session
    
    def _get_input_shape(self) -> Tuple[int, int, int]:
        """모델 입력 형태를 반환합니다."""
        input_info = self.session.get_inputs()[0]
        shape = input_info.shape
        
        # 동적 배치 크기를 1로 고정
        if shape[0] == -1:
            shape = (1,) + tuple(shape[1:])
        
        return tuple(shape)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리를 수행합니다.
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            전처리된 이미지
        """
        # 이미지 크기 조정
        input_size = self.config['input_size']
        resized = cv2.resize(image, input_size)
        
        # BGR to RGB 변환
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 -> 0-1)
        normalized = rgb.astype(np.float32) / 255.0
        
        # 차원 추가 (H, W, C) -> (1, C, H, W)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict]:
        """
        모델 출력을 후처리합니다.
        
        Args:
            outputs: 모델 출력
            original_shape: 원본 이미지 크기 (height, width)
            
        Returns:
            검출 결과 리스트
        """
        # YOLOv8 출력 처리
        predictions = outputs[0]  # (1, 84, 8400) 형태
        
        # 신뢰도 임계값 필터링
        confidence_threshold = self.config['confidence_threshold']
        
        results = []
        original_height, original_width = original_shape
        
        # 각 검출 결과 처리
        for detection in predictions[0].T:  # (8400, 84)
            # 클래스 확률 계산
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > confidence_threshold:
                # 바운딩 박스 좌표 (center_x, center_y, width, height)
                x_center, y_center, width, height = detection[:4]
                
                # 좌표 변환 (정규화된 좌표 -> 픽셀 좌표)
                x1 = int((x_center - width / 2) * original_width)
                y1 = int((y_center - height / 2) * original_height)
                x2 = int((x_center + width / 2) * original_width)
                y2 = int((y_center + height / 2) * original_height)
                
                # 결과 저장
                result = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(class_id),
                    'confidence': float(confidence),
                    'class_name': self._get_class_name(class_id)
                }
                
                results.append(result)
        
        # NMS 적용
        results = self._apply_nms(results)
        
        return results
    
    def _get_class_name(self, class_id: int) -> str:
        """클래스 ID에 해당하는 이름을 반환합니다."""
        for defect_type, info in DEFECT_TYPES.items():
            if info['id'] == class_id:
                return info['name']
        return f"unknown_{class_id}"
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Non-Maximum Suppression을 적용합니다."""
        if not detections:
            return []
        
        # 바운딩 박스와 신뢰도 추출
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # OpenCV NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config['confidence_threshold'],
            self.config['nms_threshold']
        )
        
        # 결과 필터링
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def predict(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        이미지에서 불량을 검출합니다.
        
        Args:
            image: 입력 이미지 (BGR 형식)
            
        Returns:
            (검출 결과 리스트, 추론 시간)
        """
        start_time = time.time()
        
        # 전처리
        input_tensor = self.preprocess(image)
        
        # 추론
        outputs = self.session.run(self.output_names, {self.session.get_inputs()[0].name: input_tensor})
        
        # 후처리
        original_shape = (image.shape[0], image.shape[1])
        results = self.postprocess(outputs, original_shape)
        
        inference_time = time.time() - start_time
        
        return results, inference_time
    
    def get_model_info(self) -> Dict:
        """모델 정보를 반환합니다."""
        return {
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'output_names': self.output_names,
            'device_config': self.device_config,
            'defect_types': DEFECT_TYPES,
            'config': self.config
        } 