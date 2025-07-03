#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Repository.

얼굴 데이터 저장소 인터페이스와 구현체입니다.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from domains.face_recognition.config.storage_config import get_storage_path
from typing import List, Optional, Dict, Any
import uuid

from common.logging import get_logger
from ..entities.face import Face
from ..value_objects.face_embedding import FaceEmbedding
from ..value_objects.bounding_box import BoundingBox
from ..value_objects.confidence_score import ConfidenceScore

logger = get_logger(__name__)


class FaceRepositoryInterface(ABC):
    """얼굴 저장소 인터페이스"""
    
    @abstractmethod
    def save(self, face: Face) -> bool:
        """얼굴 데이터 저장"""
        pass
    
    @abstractmethod
    def find_by_id(self, face_id: str) -> Optional[Face]:
        """ID로 얼굴 조회"""
        pass
    
    @abstractmethod
    def find_by_person_id(self, person_id: str) -> List[Face]:
        """인물 ID로 얼굴 목록 조회"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[Face]:
        """모든 얼굴 조회"""
        pass
    
    @abstractmethod
    def delete(self, face_id: str) -> bool:
        """얼굴 데이터 삭제"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """저장된 얼굴 수"""
        pass


class FaceRepository(FaceRepositoryInterface):
    """파일 기반 얼굴 저장소 구현체"""
    
    def __init__(self, storage_path: str = "data/storage/faces"):
        """
        저장소 초기화
        
        Args:
            storage_path: 얼굴 데이터 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 인덱스 파일 경로
        self.index_file = self.storage_path / "face_index.json"
        
        # 인덱스 로드 또는 생성
        self._index = self._load_index()
        
        logger.info(f"FaceRepository initialized with storage: {self.storage_path}")
    
    def save(self, face: Face) -> bool:
        """
        얼굴 데이터 저장
        
        Args:
            face: 저장할 얼굴 객체
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 얼굴 데이터를 JSON으로 직렬화
            face_data = self._serialize_face(face)
            
            # 파일에 저장
            face_file = self.storage_path / f"{face.face_id}.json"
            with open(face_file, 'w', encoding='utf-8') as f:
                json.dump(face_data, f, ensure_ascii=False, indent=2)
            
            # 인덱스 업데이트
            self._index[face.face_id] = {
                'person_id': face.person_id,
                'created_at': face.created_at.isoformat() if face.created_at else None,
                'file_path': str(face_file)
            }
            
            # 인덱스 파일 저장
            self._save_index()
            
            logger.debug(f"Face saved: {face.face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving face {face.face_id}: {str(e)}")
            return False
    
    def find_by_id(self, face_id: str) -> Optional[Face]:
        """
        ID로 얼굴 조회
        
        Args:
            face_id: 얼굴 ID
            
        Returns:
            Optional[Face]: 조회된 얼굴 (없으면 None)
        """
        try:
            if face_id not in self._index:
                return None
            
            face_file = self.storage_path / f"{face_id}.json"
            if not face_file.exists():
                logger.warning(f"Face file not found: {face_file}")
                return None
            
            with open(face_file, 'r', encoding='utf-8') as f:
                face_data = json.load(f)
            
            face = self._deserialize_face(face_data)
            return face
            
        except Exception as e:
            logger.error(f"Error loading face {face_id}: {str(e)}")
            return None
    
    def find_by_person_id(self, person_id: str) -> List[Face]:
        """
        인물 ID로 얼굴 목록 조회
        
        Args:
            person_id: 인물 ID
            
        Returns:
            List[Face]: 해당 인물의 얼굴 목록
        """
        faces = []
        
        try:
            for face_id, index_data in self._index.items():
                if index_data.get('person_id') == person_id:
                    face = self.find_by_id(face_id)
                    if face:
                        faces.append(face)
            
            logger.debug(f"Found {len(faces)} faces for person {person_id}")
            return faces
            
        except Exception as e:
            logger.error(f"Error finding faces for person {person_id}: {str(e)}")
            return []
    
    def find_all(self) -> List[Face]:
        """
        모든 얼굴 조회
        
        Returns:
            List[Face]: 저장된 모든 얼굴 목록
        """
        faces = []
        
        try:
            for face_id in self._index.keys():
                face = self.find_by_id(face_id)
                if face:
                    faces.append(face)
            
            logger.debug(f"Found {len(faces)} total faces")
            return faces
            
        except Exception as e:
            logger.error(f"Error loading all faces: {str(e)}")
            return []
    
    def delete(self, face_id: str) -> bool:
        """
        얼굴 데이터 삭제
        
        Args:
            face_id: 삭제할 얼굴 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if face_id not in self._index:
                logger.warning(f"Face not found for deletion: {face_id}")
                return False
            
            # 파일 삭제
            face_file = self.storage_path / f"{face_id}.json"
            if face_file.exists():
                face_file.unlink()
            
            # 인덱스에서 제거
            del self._index[face_id]
            
            # 인덱스 파일 저장
            self._save_index()
            
            logger.info(f"Face deleted: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting face {face_id}: {str(e)}")
            return False
    
    def count(self) -> int:
        """
        저장된 얼굴 수
        
        Returns:
            int: 얼굴 수
        """
        return len(self._index)
    
    def _serialize_face(self, face: Face) -> Dict[str, Any]:
        """얼굴 객체를 JSON 직렬화 가능한 딕셔너리로 변환"""
        data = {
            'face_id': face.face_id,
            'person_id': face.person_id,
            'confidence': face.confidence,
            'bbox': face.bbox,
            'landmarks': face.landmarks,
            'created_at': face.created_at.isoformat() if face.created_at else None
        }
        
        # NumPy 배열을 리스트로 변환
        if hasattr(face.embedding, 'tolist'):
            data['embedding'] = face.embedding.tolist()
        else:
            data['embedding'] = list(face.embedding)
        
        return data
    
    def _deserialize_face(self, data: Dict[str, Any]) -> Face:
        """JSON 데이터를 얼굴 객체로 변환"""
        from datetime import datetime
        import numpy as np
        
        # 날짜 복원
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        # NumPy 배열 복원
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        
        return Face(
            face_id=data['face_id'],
            person_id=data.get('person_id'),
            embedding=embedding,
            confidence=data.get('confidence', 0.0),
            bbox=data.get('bbox', [0, 0, 0, 0]),
            landmarks=data.get('landmarks'),
            created_at=created_at
        )
    
    def _load_index(self) -> Dict[str, Any]:
        """인덱스 파일 로드"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading index file: {str(e)}")
        
        return {}
    
    def _save_index(self):
        """인덱스 파일 저장"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving index file: {str(e)}") 