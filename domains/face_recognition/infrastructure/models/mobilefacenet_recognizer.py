#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MobileFaceNet 인식기

ONNX Runtime을 사용하여 MobileFaceNet 모델로 얼굴 임베딩을 생성합니다.
1차 선택 인식 모델로 사용됩니다.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
import logging

from shared.vision_core.recognition.base_recognizer import BaseRecognizer
from domains.face_recognition.core.value_objects.face_embedding import FaceEmbedding

logger = logging.getLogger(__name__)

class MobileFaceNetRecognizer(BaseRecognizer):
    """MobileFaceNet 기반 얼굴 인식기"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        MobileFaceNet 인식기 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
            device: 실행 디바이스 (auto, cpu, cuda)
        """
        # BaseRecognizer 초기화 (필수 인자 전달)
        super().__init__(
            model_path=model_path,
            input_size=(112, 112),
            embedding_size=192,
            use_gpu=(device == "cuda")
        )
        
        self.device = self._select_device(device)
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self.std = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        
        # ONNX Runtime 세션 초기화
        self.session = self._initialize_session()
        
        logger.info(f"MobileFaceNet recognizer initialized with device: {self.device}")
    
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
                        "gpu_mem_limit": 1 * 1024 * 1024 * 1024,  # 1GB
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
    
    def load_model(self, model_path: str) -> bool:
        """
        모델 로드 (BaseRecognizer 추상 메서드 구현)
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            로드 성공 여부
        """
        try:
            self.model_path = model_path
            self.session = self._initialize_session()
            logger.info(f"Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지 전처리
        
        Args:
            face_image: 얼굴 이미지 (BGR)
            
        Returns:
            전처리된 이미지 텐서 (batch, 3, 112, 112)
        """
        # 얼굴 정렬 (랜드마크가 있는 경우)
        aligned_face = self._align_face(face_image)
        
        # 이미지 리사이즈
        resized_face = cv2.resize(aligned_face, self.input_size)
        
        # BGR to RGB 변환
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        
        # 정규화
        normalized_face = (rgb_face.astype(np.float32) - self.mean) / self.std
        
        # 차원 변환: (112, 112, 3) -> (3, 112, 112)
        transposed_face = np.transpose(normalized_face, (2, 0, 1))
        
        # 배치 차원 추가: (3, 112, 112) -> (1, 3, 112, 112)
        input_tensor = np.expand_dims(transposed_face, axis=0)
        
        return input_tensor
    
    def _align_face(self, face_image: np.ndarray, landmarks: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        얼굴 정렬
        
        Args:
            face_image: 얼굴 이미지
            landmarks: 랜드마크 좌표 (선택사항)
            
        Returns:
            정렬된 얼굴 이미지
        """
        if landmarks is not None and len(landmarks) >= 5:
            # 랜드마크 기반 정렬
            return self._align_face_with_landmarks(face_image, landmarks)
        else:
            # 기본 정렬 (중앙 크롭)
            return self._center_crop_face(face_image)
    
    def _align_face_with_landmarks(self, 
                                  face_image: np.ndarray, 
                                  landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """랜드마크 기반 얼굴 정렬"""
        try:
            # 눈 좌표 추출 (첫 번째와 두 번째 랜드마크)
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # 눈 사이의 각도 계산
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            # 회전 중심점 (눈의 중점)
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            
            # 회전 행렬 생성
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), eye_angle, 1.0)
            
            # 이미지 회전
            aligned_face = cv2.warpAffine(face_image, rotation_matrix, 
                                        (face_image.shape[1], face_image.shape[0]))
            
            return aligned_face
            
        except Exception as e:
            logger.warning(f"Landmark-based alignment failed: {str(e)}, using center crop")
            return self._center_crop_face(face_image)
    
    def _center_crop_face(self, face_image: np.ndarray) -> np.ndarray:
        """중앙 크롭 기반 얼굴 정렬"""
        height, width = face_image.shape[:2]
        
        # 정사각형 크롭
        size = min(height, width)
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        
        cropped_face = face_image[start_y:start_y + size, start_x:start_x + size]
        
        return cropped_face
    
    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        모델 출력 후처리
        
        Args:
            output: 모델 출력 (임베딩 벡터)
            
        Returns:
            정규화된 임베딩 벡터
        """
        # 배치 차원 제거
        embedding = output[0]
        
        # L2 정규화
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def extract_embedding(self, face_image: np.ndarray, 
                         landmarks: Optional[List[Tuple[int, int]]] = None) -> FaceEmbedding:
        """
        얼굴 이미지에서 임베딩 추출
        
        Args:
            face_image: 얼굴 이미지 (BGR)
            landmarks: 랜드마크 좌표 (선택사항)
            
        Returns:
            얼굴 임베딩 객체
        """
        try:
            # 전처리
            input_tensor = self.preprocess(face_image)
            
            # 추론 실행
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_tensor})
            
            # 후처리
            embedding_vector = self.postprocess(output[0])
            
            # FaceEmbedding 객체 생성
            face_embedding = FaceEmbedding(
                vector=embedding_vector,
                model_name="mobilefacenet_recognizer"
            )
            
            logger.debug(f"Extracted embedding with size: {len(embedding_vector)}")
            return face_embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {str(e)}")
            raise
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray],
                               landmarks_list: Optional[List[List[Tuple[int, int]]]] = None) -> List[FaceEmbedding]:
        """
        배치 얼굴 이미지에서 임베딩 추출
        
        Args:
            face_images: 얼굴 이미지 리스트
            landmarks_list: 랜드마크 좌표 리스트 (선택사항)
            
        Returns:
            얼굴 임베딩 객체 리스트
        """
        embeddings = []
        
        for i, face_image in enumerate(face_images):
            landmarks = None
            if landmarks_list and i < len(landmarks_list):
                landmarks = landmarks_list[i]
            
            try:
                embedding = self.extract_embedding(face_image, landmarks)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to extract embedding for image {i}: {str(e)}")
                # 실패한 경우 None 추가
                embeddings.append(None)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: FaceEmbedding, 
                           embedding2: FaceEmbedding) -> float:
        """
        두 임베딩 간의 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            
        Returns:
            코사인 유사도 (0.0 ~ 1.0)
        """
        try:
            # 코사인 유사도 계산
            dot_product = np.dot(embedding1.vector, embedding2.vector)
            norm1 = np.linalg.norm(embedding1.vector)
            norm2 = np.linalg.norm(embedding2.vector)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # 유사도를 0~1 범위로 클리핑
            similarity = np.clip(similarity, -1.0, 1.0)
            
            # 코사인 유사도를 거리로 변환 (0~2 범위)
            distance = 1.0 - similarity
            
            return distance
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return float('inf')  # 최대 거리 반환
    
    def find_best_match(self, query_embedding: FaceEmbedding,
                       candidate_embeddings: List[FaceEmbedding],
                       threshold: float = 0.6) -> Tuple[Optional[FaceEmbedding], float]:
        """
        최적 매치 찾기
        
        Args:
            query_embedding: 쿼리 임베딩
            candidate_embeddings: 후보 임베딩 리스트
            threshold: 매칭 임계값
            
        Returns:
            (최적 매치 임베딩, 최소 거리)
        """
        if not candidate_embeddings:
            return None, float('inf')
        
        best_match = None
        min_distance = float('inf')
        
        for candidate in candidate_embeddings:
            if candidate is None:
                continue
                
            distance = self.calculate_similarity(query_embedding, candidate)
            
            if distance < min_distance:
                min_distance = distance
                best_match = candidate
        
        # 임계값 확인
        if min_distance > threshold:
            return None, min_distance
        
        return best_match, min_distance
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "name": "MobileFaceNet",
            "type": "recognition",
            "engine": "onnxruntime",
            "device": self.device,
            "input_size": self.input_size,
            "embedding_size": self.embedding_size,
            "model_path": self.model_path
        }
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        if hasattr(self, 'session'):
            del self.session 