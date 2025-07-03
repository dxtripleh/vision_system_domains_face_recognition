#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Service.

얼굴 인식을 위한 도메인 서비스입니다.
"""

import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2

from common.logging import get_logger
from ..entities.face import Face
from ..entities.person import Person
from ..value_objects.face_embedding import FaceEmbedding
from ..value_objects.bounding_box import BoundingBox
from ..value_objects.confidence_score import ConfidenceScore
from ..repositories.face_repository import FaceRepositoryInterface
from ..repositories.person_repository import PersonRepositoryInterface

logger = get_logger(__name__)


class FaceRecognitionService:
    """얼굴 인식 서비스"""
    
    def __init__(self, 
                 recognizer=None, 
                 person_repository: Optional[PersonRepositoryInterface] = None, 
                 face_repository: Optional[FaceRepositoryInterface] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 use_mock: bool = False):
        """
        서비스 초기화
        
        Args:
            recognizer: 얼굴 인식기 (의존성 주입)
            person_repository: 인물 저장소 (의존성 주입)
            face_repository: 얼굴 저장소 (의존성 주입)
            config: 설정 딕셔너리
            use_mock: Mock 구현 사용 여부 (테스트용)
        """
        self.use_mock = use_mock
        self.recognizer = recognizer
        self.person_repository = person_repository
        self.face_repository = face_repository
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        self.embedding_dimension = self.config.get('embedding_dimension', 512)
        
        # Repository가 주입되지 않았다면 기본 구현체 사용
        if self.person_repository is None:
            from ..repositories.person_repository import PersonRepository
            storage_path = "data/test/persons" if use_mock else "data/storage/persons"
            self.person_repository = PersonRepository(storage_path)
        
        if self.face_repository is None:
            from ..repositories.face_repository import FaceRepository
            storage_path = "data/test/faces" if use_mock else "data/storage/faces"
            self.face_repository = FaceRepository(storage_path)
        
        # 인식기가 주입되지 않았다면 기본 ArcFace 사용 (Mock 모드가 아닌 경우)
        if self.recognizer is None and not use_mock:
            self._initialize_default_recognizer()
        
        logger.info(f"FaceRecognitionService initialized with config: {self.config}")
    
    def _initialize_default_recognizer(self):
        """기본 ArcFace 인식기 초기화"""
        try:
            from ...infrastructure.models.arcface_recognizer import ArcFaceRecognizer
            
            model_path = self.config.get('model_path', 'models/weights/face_recognition_arcface_r50_20250628.onnx')
            self.recognizer = ArcFaceRecognizer(
                model_path=model_path,
                input_size=self.config.get('input_size', (112, 112)),
                use_gpu=self.config.get('use_gpu', True)
            )
            
            if self.recognizer.load_model():
                logger.info("Default ArcFace recognizer loaded successfully")
            else:
                logger.warning("Failed to load ArcFace recognizer, using mock implementation")
                self.recognizer = None
                
        except Exception as e:
            logger.warning(f"Error initializing ArcFace recognizer: {str(e)}, using mock implementation")
            self.recognizer = None
    
    def extract_embedding(self, face_image: np.ndarray, model_name: str = "arcface") -> FaceEmbedding:
        """
        얼굴 이미지에서 임베딩을 추출합니다.
        
        Args:
            face_image: 얼굴 이미지 (BGR 형식)
            model_name: 사용할 모델명
            
        Returns:
            FaceEmbedding: 얼굴 임베딩
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("유효하지 않은 얼굴 이미지입니다")
        
        logger.debug(f"임베딩 추출 시작 - 모델: {model_name}")
        
        try:
            # 이미지 전처리
            processed_image = self._preprocess_face_image(face_image)
            
            # 임베딩 추출
            if self.recognizer is not None:
                # 실제 AI 모델 사용
                embedding = self.recognizer.extract_embedding(processed_image)
                if embedding is None:
                    logger.warning("AI 모델에서 임베딩 추출 실패, Mock 임베딩 사용")
                    embedding_vector = self._extract_embedding_mock(processed_image)
                    embedding = FaceEmbedding(
                        vector=embedding_vector,
                        model_name=model_name
                    )
            else:
                # Mock 구현 사용 (모델이 없는 경우)
                embedding_vector = self._extract_embedding_mock(processed_image)
                embedding = FaceEmbedding(
                    vector=embedding_vector,
                    model_name=model_name
                )
                logger.warning("AI 모델이 없어 Mock 임베딩을 사용합니다")
            
            logger.debug(f"임베딩 추출 완료 - 차원: {embedding.dimension}, 모델: {model_name}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 추출 중 오류 발생: {str(e)}")
            raise
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray], 
                                model_name: str = "arcface") -> List[FaceEmbedding]:
        """
        여러 얼굴 이미지에서 배치로 임베딩을 추출합니다.
        
        Args:
            face_images: 얼굴 이미지 리스트
            model_name: 사용할 모델명
            
        Returns:
            List[FaceEmbedding]: 임베딩 리스트
        """
        if not face_images:
            return []
        
        embeddings = []
        for i, face_image in enumerate(face_images):
            try:
                embedding = self.extract_embedding(face_image, model_name)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"배치 임베딩 추출 중 오류 (이미지 {i}): {str(e)}")
                # 오류가 발생한 경우 None 대신 빈 임베딩 추가
                embeddings.append(None)
        
        return embeddings
    
    def identify_face(self, face: Face) -> Optional[Person]:
        """
        얼굴을 식별하여 등록된 인물을 찾습니다.
        
        Args:
            face: 식별할 얼굴
            
        Returns:
            Optional[Person]: 매칭된 인물 (없으면 None)
        """
        if face.embedding is None:
            logger.warning("얼굴에 임베딩이 없어 식별할 수 없습니다")
            return None
        
        logger.debug(f"얼굴 식별 시작 - Face ID: {face.face_id}")
        
        try:
            # 등록된 모든 인물 조회
            registered_persons = self.person_repository.find_all()
            
            best_match = None
            best_similarity = 0.0
            
            for person in registered_persons:
                # 각 인물의 평균 임베딩과 비교
                avg_embedding = person.get_average_embedding()
                if avg_embedding is None:
                    continue
                
                similarity = face.embedding.similarity_with(avg_embedding, 'cosine')
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = person
            
            if best_match:
                logger.info(f"얼굴 식별 성공 - 인물: {best_match.name}, 유사도: {best_similarity:.3f}")
            else:
                logger.debug("매칭되는 인물을 찾지 못했습니다")
            
            return best_match
            
        except Exception as e:
            logger.error(f"얼굴 식별 중 오류 발생: {str(e)}")
            return None
    
    def verify_face(self, face: Face, person_id: str) -> Tuple[bool, float]:
        """
        얼굴이 특정 인물과 일치하는지 검증합니다.
        
        Args:
            face: 검증할 얼굴
            person_id: 인물 ID
            
        Returns:
            Tuple[bool, float]: (일치 여부, 유사도 점수)
        """
        if face.embedding is None:
            return False, 0.0
        
        logger.debug(f"얼굴 검증 시작 - Face ID: {face.face_id}, Person ID: {person_id}")
        
        try:
            # 인물 조회
            person = self.person_repository.find_by_id(person_id)
            if person is None:
                logger.warning(f"인물을 찾을 수 없습니다: {person_id}")
                return False, 0.0
            
            # 평균 임베딩과 비교
            avg_embedding = person.get_average_embedding()
            if avg_embedding is None:
                logger.warning(f"인물의 임베딩이 없습니다: {person_id}")
                return False, 0.0
            
            similarity = face.embedding.similarity_with(avg_embedding, 'cosine')
            is_match = similarity >= self.similarity_threshold
            
            logger.debug(f"얼굴 검증 완료 - 결과: {is_match}, 유사도: {similarity:.3f}")
            
            return is_match, similarity
            
        except Exception as e:
            logger.error(f"얼굴 검증 중 오류 발생: {str(e)}")
            return False, 0.0
    
    def register_person(self, name: str, face_images: List[np.ndarray], 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        새로운 인물을 등록합니다.
        
        Args:
            name: 인물 이름
            face_images: 얼굴 이미지 리스트
            metadata: 추가 메타데이터
            
        Returns:
            str: 생성된 인물 ID
            
        Raises:
            ValueError: 얼굴 이미지가 없거나 임베딩 추출에 실패한 경우
        """
        if not face_images:
            raise ValueError("최소 하나의 얼굴 이미지가 필요합니다")
        
        logger.info(f"인물 등록 시작 - 이름: {name}, 이미지 수: {len(face_images)}")
        
        try:
            # 인물 ID 생성
            person_id = str(uuid.uuid4())
            current_time = time.time()
            
            # 얼굴 임베딩 추출 및 저장
            face_entities = []
            for i, face_image in enumerate(face_images):
                try:
                    # 임베딩 추출
                    embedding = self.extract_embedding(face_image)
                    
                    # Face 엔티티 생성
                    face = Face(
                        face_id=str(uuid.uuid4()),
                        person_id=person_id,
                        embedding=embedding.vector,  # FaceEmbedding에서 벡터 추출
                        confidence=0.9,  # float 값으로 직접 전달
                        bbox=[0, 0, face_image.shape[1], face_image.shape[0]]  # List[float]로 전달
                    )
                    
                    # 얼굴 저장
                    if self.face_repository.save(face):
                        face_entities.append(face)
                        logger.debug(f"얼굴 {i+1}/{len(face_images)} 저장 완료")
                    else:
                        logger.warning(f"얼굴 {i+1}/{len(face_images)} 저장 실패")
                        
                except Exception as e:
                    logger.error(f"얼굴 이미지 {i+1} 처리 중 오류: {str(e)}")
                    continue
            
            if not face_entities:
                raise ValueError("모든 얼굴 이미지 처리에 실패했습니다")
            
            # Person 엔티티 생성
            person = Person(
                person_id=person_id,
                name=name,
                face_embeddings=[face.embedding for face in face_entities],  # 임베딩만 저장
                metadata=metadata or {}
            )
            
            # 인물 저장
            if self.person_repository.save(person):
                logger.info(f"인물 등록 완료 - ID: {person_id}, 이름: {name}, 얼굴 수: {len(face_entities)}")
                return person_id
            else:
                raise RuntimeError("인물 저장에 실패했습니다")
                
        except Exception as e:
            logger.error(f"인물 등록 중 오류 발생: {str(e)}")
            raise
    
    def update_person(self, person_id: str, name: Optional[str] = None, 
                     additional_face_images: Optional[List[np.ndarray]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        등록된 인물 정보를 업데이트합니다.
        
        Args:
            person_id: 인물 ID
            name: 새로운 이름 (변경하지 않으려면 None)
            additional_face_images: 추가할 얼굴 이미지들
            metadata: 업데이트할 메타데이터
            
        Returns:
            bool: 업데이트 성공 여부
        """
        logger.info(f"인물 업데이트 시작 - ID: {person_id}")
        
        try:
            # 기존 인물 조회
            person = self.person_repository.find_by_id(person_id)
            if person is None:
                logger.warning(f"인물을 찾을 수 없습니다: {person_id}")
                return False
            
            # 이름 업데이트
            if name is not None:
                person.name = name
                logger.debug(f"이름 업데이트: {person.name} -> {name}")
            
            # 메타데이터 업데이트
            if metadata is not None:
                person.metadata.update(metadata)
                logger.debug("메타데이터 업데이트 완료")
            
            # 추가 얼굴 이미지 처리
            if additional_face_images:
                current_time = time.time()
                
                for i, face_image in enumerate(additional_face_images):
                    try:
                        # 임베딩 추출
                        embedding = self.extract_embedding(face_image)
                        
                        # Face 엔티티 생성
                        face = Face(
                            face_id=str(uuid.uuid4()),
                            person_id=person_id,
                            embedding=embedding.vector,  # FaceEmbedding에서 벡터 추출
                            confidence=0.9,  # float 값으로 직접 전달
                            bbox=[0, 0, face_image.shape[1], face_image.shape[0]]  # List[float]로 전달
                        )
                        
                        # 얼굴 저장
                        if self.face_repository.save(face):
                            person.faces.append(face)
                            logger.debug(f"추가 얼굴 {i+1}/{len(additional_face_images)} 저장 완료")
                        
                    except Exception as e:
                        logger.error(f"추가 얼굴 이미지 {i+1} 처리 중 오류: {str(e)}")
                        continue
            
            # 인물 정보 저장
            if self.person_repository.save(person):
                logger.info(f"인물 업데이트 완료 - ID: {person_id}")
                return True
            else:
                logger.error(f"인물 업데이트 저장 실패 - ID: {person_id}")
                return False
                
        except Exception as e:
            logger.error(f"인물 업데이트 중 오류 발생: {str(e)}")
            return False
    
    def remove_person(self, person_id: str) -> bool:
        """
        등록된 인물을 삭제합니다.
        
        Args:
            person_id: 삭제할 인물 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        logger.info(f"인물 삭제 시작 - ID: {person_id}")
        
        try:
            # 인물의 모든 얼굴 데이터 삭제
            faces = self.face_repository.find_by_person_id(person_id)
            deleted_faces = 0
            
            for face in faces:
                if self.face_repository.delete(face.face_id):
                    deleted_faces += 1
            
            logger.debug(f"얼굴 데이터 {deleted_faces}개 삭제 완료")
            
            # 인물 데이터 삭제
            if self.person_repository.delete(person_id):
                logger.info(f"인물 삭제 완료 - ID: {person_id}")
                return True
            else:
                logger.error(f"인물 삭제 실패 - ID: {person_id}")
                return False
                
        except Exception as e:
            logger.error(f"인물 삭제 중 오류 발생: {str(e)}")
            return False
    
    def get_all_persons(self) -> List[Person]:
        """
        등록된 모든 인물 목록을 조회합니다.
        
        Returns:
            List[Person]: 인물 목록
        """
        try:
            persons = self.person_repository.find_all()
            logger.debug(f"등록된 인물 수: {len(persons)}")
            return persons
        except Exception as e:
            logger.error(f"인물 목록 조회 중 오류 발생: {str(e)}")
            return []
    
    def get_person_by_id(self, person_id: str) -> Optional[Person]:
        """
        특정 인물 정보를 조회합니다.
        
        Args:
            person_id: 인물 ID
            
        Returns:
            Optional[Person]: 인물 정보 (없으면 None)
        """
        try:
            person = self.person_repository.find_by_id(person_id)
            if person:
                logger.debug(f"인물 조회 성공 - ID: {person_id}, 이름: {person.name}")
            else:
                logger.debug(f"인물을 찾을 수 없음 - ID: {person_id}")
            return person
        except Exception as e:
            logger.error(f"인물 조회 중 오류 발생: {str(e)}")
            return None
    
    def _preprocess_face_image(self, face_image: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지 전처리
        
        Args:
            face_image: 입력 얼굴 이미지
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        # 기본 전처리: 크기 조정 및 정규화
        if face_image.shape[:2] != (112, 112):
            processed = cv2.resize(face_image, (112, 112))
        else:
            processed = face_image.copy()
        
        return processed
    
    def _extract_embedding_mock(self, face_image: np.ndarray) -> np.ndarray:
        """
        Mock 임베딩 추출 (AI 모델이 없을 때 사용)
        
        Args:
            face_image: 얼굴 이미지
            
        Returns:
            np.ndarray: Mock 임베딩 벡터
        """
        # 이미지 기반 의사 임베딩 생성
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8))
        flattened = resized.flatten().astype(np.float32)
        
        # 512차원으로 확장
        mock_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
        mock_embedding[:len(flattened)] = flattened
        
        # 정규화
        norm = np.linalg.norm(mock_embedding)
        if norm > 0:
            mock_embedding = mock_embedding / norm
        
        return mock_embedding
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        서비스 통계 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        try:
            total_persons = self.person_repository.count()
            total_faces = self.face_repository.count()
            
            return {
                'total_persons': total_persons,
                'total_faces': total_faces,
                'average_faces_per_person': total_faces / total_persons if total_persons > 0 else 0,
                'similarity_threshold': self.similarity_threshold,
                'embedding_dimension': self.embedding_dimension,
                'recognizer_available': self.recognizer is not None
            }
        except Exception as e:
            logger.error(f"통계 정보 생성 중 오류 발생: {str(e)}")
            return {} 