#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RetinaFace MobileNet0.25 검출기

ONNX Runtime을 사용하여 RetinaFace MobileNet0.25 모델로 얼굴을 검출합니다.
1차 선택 모델로 사용됩니다.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
import logging

from shared.vision_core.detection.base_detector import BaseDetector
from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore

logger = logging.getLogger(__name__)

class RetinaFaceDetector(BaseDetector):
    """RetinaFace MobileNet0.25 기반 얼굴 검출기"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        RetinaFace 검출기 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
            device: 실행 디바이스 (auto, cpu, cuda)
        """
        super().__init__()
        
        self.model_path = model_path
        self.device = self._select_device(device)
        self.input_size = (640, 640)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.std = np.array([1, 1, 1], dtype=np.float32)
        
        # ONNX Runtime 세션 초기화
        self.session = self._initialize_session()
        
        logger.info(f"RetinaFace detector initialized with device: {self.device}")
    
    def _select_device(self, device: str) -> str:
        """실행 디바이스 선택"""
        if device == "auto":
            # GPU 사용 가능 여부 확인
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_session(self) -> ort.InferenceSession:
        """ONNX Runtime 세션 초기화"""
        try:
            # 실행 프로바이더 설정
            providers = []
            if self.device == "cuda":
                providers = [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                        "cudnn_conv_use_max_workspace": "1",
                        "do_copy_in_default_stream": "1",
                    }),
                    "CPUExecutionProvider"
                ]
            else:
                providers = ["CPUExecutionProvider"]
            
            # 세션 옵션 설정
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_reuse = True
            
            # 세션 생성
            session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"ONNX Runtime session initialized with providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {str(e)}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            전처리된 이미지, 스케일 비율, 원본 크기
        """
        # 원본 크기 저장
        original_height, original_width = image.shape[:2]
        
        # 이미지 리사이즈
        resized_image = cv2.resize(image, self.input_size)
        
        # BGR to RGB 변환
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 정규화
        normalized_image = (rgb_image.astype(np.float32) - self.mean) / self.std
        
        # 차원 추가 (batch dimension)
        input_tensor = np.expand_dims(normalized_image, axis=0)
        
        # 스케일 비율 계산
        scale_x = original_width / self.input_size[0]
        scale_y = original_height / self.input_size[1]
        
        return input_tensor, (scale_x, scale_y), (original_width, original_height)
    
    def postprocess(self, 
                   outputs: List[np.ndarray], 
                   scale: Tuple[float, float],
                   original_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        모델 출력 후처리
        
        Args:
            outputs: 모델 출력
            scale: 스케일 비율 (scale_x, scale_y)
            original_size: 원본 이미지 크기 (width, height)
            
        Returns:
            검출된 얼굴들의 바운딩 박스 리스트
        """
        scale_x, scale_y = scale
        original_width, original_height = original_size
        
        # RetinaFace 출력 파싱
        # outputs[0]: bbox predictions
        # outputs[1]: confidence scores
        # outputs[2]: landmark predictions (if available)
        
        bbox_predictions = outputs[0][0]  # (N, 4)
        confidence_scores = outputs[1][0]  # (N,)
        
        if len(outputs) > 2:
            landmark_predictions = outputs[2][0]  # (N, 10) - 5 landmarks * 2 coordinates
        else:
            landmark_predictions = None
        
        bounding_boxes = []
        
        for i in range(len(bbox_predictions)):
            confidence = confidence_scores[i]
            
            # 신뢰도 임계값 확인
            if confidence < self.confidence_threshold:
                continue
            
            # 바운딩 박스 좌표 추출
            x1, y1, x2, y2 = bbox_predictions[i]
            
            # 원본 이미지 좌표로 변환
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # 경계 확인
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))
            
            # 유효한 바운딩 박스인지 확인
            if x2 <= x1 or y2 <= y1:
                continue
            
            # BoundingBox 객체 생성
            bbox = BoundingBox(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1
            )
            
            # ConfidenceScore 객체 생성
            confidence_score = ConfidenceScore(value=float(confidence))
            
            # 랜드마크 정보 추가 (있는 경우)
            landmarks = None
            if landmark_predictions is not None:
                landmarks = self._extract_landmarks(
                    landmark_predictions[i], scale, original_size
                )
            
            # 메타데이터 설정
            bbox.metadata = {
                "confidence": confidence_score,
                "model": "retinaface_mobilenet025",
                "landmarks": landmarks
            }
            
            bounding_boxes.append(bbox)
        
        # NMS 적용
        bounding_boxes = self._apply_nms(bounding_boxes)
        
        return bounding_boxes
    
    def _extract_landmarks(self, 
                          landmark_pred: np.ndarray, 
                          scale: Tuple[float, float],
                          original_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """랜드마크 좌표 추출"""
        scale_x, scale_y = scale
        landmarks = []
        
        # 5개 랜드마크 (왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입, 오른쪽 입)
        for i in range(0, 10, 2):
            x = int(landmark_pred[i] * scale_x)
            y = int(landmark_pred[i + 1] * scale_y)
            landmarks.append((x, y))
        
        return landmarks
    
    def _apply_nms(self, bounding_boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Non-Maximum Suppression 적용"""
        if not bounding_boxes:
            return []
        
        # 신뢰도 기준으로 정렬
        sorted_boxes = sorted(bounding_boxes, 
                            key=lambda bbox: bbox.metadata["confidence"].value,
                            reverse=True)
        
        selected_boxes = []
        
        while sorted_boxes:
            # 가장 높은 신뢰도를 가진 박스 선택
            current_box = sorted_boxes.pop(0)
            selected_boxes.append(current_box)
            
            # 나머지 박스들과 IoU 계산
            remaining_boxes = []
            for box in sorted_boxes:
                iou = self._calculate_iou(current_box, box)
                if iou < self.nms_threshold:
                    remaining_boxes.append(box)
            
            sorted_boxes = remaining_boxes
        
        return selected_boxes
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """IoU (Intersection over Union) 계산"""
        # 교집합 영역 계산
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 합집합 영역 계산
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        이미지에서 얼굴 검출
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            검출된 얼굴들의 바운딩 박스 리스트
        """
        try:
            # 전처리
            input_tensor, scale, original_size = self.preprocess(image)
            
            # 추론 실행
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # 후처리
            bounding_boxes = self.postprocess(outputs, scale, original_size)
            
            logger.debug(f"Detected {len(bounding_boxes)} faces using RetinaFace")
            return bounding_boxes
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        if hasattr(self, 'session'):
            del self.session 