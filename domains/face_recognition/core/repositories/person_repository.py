#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Person Repository.

인물 데이터 저장소 인터페이스와 구현체입니다.
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
from ..entities.person import Person
from ..entities.face import Face
from ..value_objects.face_embedding import FaceEmbedding

logger = get_logger(__name__)


class PersonRepositoryInterface(ABC):
    """인물 저장소 인터페이스"""
    
    @abstractmethod
    def save(self, person: Person) -> bool:
        """인물 데이터 저장"""
        pass
    
    @abstractmethod
    def find_by_id(self, person_id: str) -> Optional[Person]:
        """ID로 인물 조회"""
        pass
    
    @abstractmethod
    def find_by_name(self, name: str) -> List[Person]:
        """이름으로 인물 조회"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[Person]:
        """모든 인물 조회"""
        pass
    
    @abstractmethod
    def delete(self, person_id: str) -> bool:
        """인물 데이터 삭제"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """저장된 인물 수"""
        pass
    
    @abstractmethod
    def search(self, query: str) -> List[Person]:
        """인물 검색"""
        pass


class PersonRepository(PersonRepositoryInterface):
    """파일 기반 인물 저장소 구현체"""
    
    def __init__(self, storage_path: str = "data/storage/persons"):
        """
        저장소 초기화
        
        Args:
            storage_path: 인물 데이터 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 인덱스 파일 경로
        self.index_file = self.storage_path / "person_index.json"
        
        # 인덱스 로드 또는 생성
        self._index = self._load_index()
        
        logger.info(f"PersonRepository initialized with storage: {self.storage_path}")
    
    def save(self, person: Person) -> bool:
        """
        인물 데이터 저장
        
        Args:
            person: 저장할 인물 객체
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 인물 데이터를 JSON으로 직렬화
            person_data = self._serialize_person(person)
            
            # 파일에 저장
            person_file = self.storage_path / f"{person.person_id}.json"
            with open(person_file, 'w', encoding='utf-8') as f:
                json.dump(person_data, f, ensure_ascii=False, indent=2)
            
            # 인덱스 업데이트
            self._index[person.person_id] = {
                'name': person.name,
                'created_at': person.created_at.isoformat() if person.created_at else None,
                'updated_at': person.updated_at.isoformat() if person.updated_at else None,
                'face_count': len(person.face_embeddings),
                'file_path': str(person_file)
            }
            
            # 인덱스 파일 저장
            self._save_index()
            
            logger.debug(f"Person saved: {person.person_id} ({person.name})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving person {person.person_id}: {str(e)}")
            return False
    
    def find_by_id(self, person_id: str) -> Optional[Person]:
        """
        ID로 인물 조회
        
        Args:
            person_id: 인물 ID
            
        Returns:
            Optional[Person]: 조회된 인물 (없으면 None)
        """
        try:
            if person_id not in self._index:
                return None
            
            person_file = self.storage_path / f"{person_id}.json"
            if not person_file.exists():
                logger.warning(f"Person file not found: {person_file}")
                return None
            
            with open(person_file, 'r', encoding='utf-8') as f:
                person_data = json.load(f)
            
            person = self._deserialize_person(person_data)
            return person
            
        except Exception as e:
            logger.error(f"Error loading person {person_id}: {str(e)}")
            return None
    
    def find_by_name(self, name: str) -> List[Person]:
        """
        이름으로 인물 조회
        
        Args:
            name: 인물 이름
            
        Returns:
            List[Person]: 해당 이름의 인물 목록
        """
        persons = []
        
        try:
            for person_id, index_data in self._index.items():
                if index_data.get('name', '').lower() == name.lower():
                    person = self.find_by_id(person_id)
                    if person:
                        persons.append(person)
            
            logger.debug(f"Found {len(persons)} persons with name '{name}'")
            return persons
            
        except Exception as e:
            logger.error(f"Error finding persons by name '{name}': {str(e)}")
            return []
    
    def find_all(self) -> List[Person]:
        """
        모든 인물 조회
        
        Returns:
            List[Person]: 저장된 모든 인물 목록
        """
        persons = []
        
        try:
            for person_id in self._index.keys():
                person = self.find_by_id(person_id)
                if person:
                    persons.append(person)
            
            logger.debug(f"Found {len(persons)} total persons")
            return persons
            
        except Exception as e:
            logger.error(f"Error loading all persons: {str(e)}")
            return []
    
    def delete(self, person_id: str) -> bool:
        """
        인물 데이터 삭제
        
        Args:
            person_id: 삭제할 인물 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if person_id not in self._index:
                logger.warning(f"Person not found for deletion: {person_id}")
                return False
            
            # 파일 삭제
            person_file = self.storage_path / f"{person_id}.json"
            if person_file.exists():
                person_file.unlink()
            
            # 인덱스에서 제거
            del self._index[person_id]
            
            # 인덱스 파일 저장
            self._save_index()
            
            logger.info(f"Person deleted: {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting person {person_id}: {str(e)}")
            return False
    
    def count(self) -> int:
        """
        저장된 인물 수
        
        Returns:
            int: 인물 수
        """
        return len(self._index)
    
    def search(self, query: str) -> List[Person]:
        """
        인물 검색 (이름, 메타데이터 기반)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            List[Person]: 검색 결과 인물 목록
        """
        persons = []
        query_lower = query.lower()
        
        try:
            for person_id, index_data in self._index.items():
                # 이름 검색
                name = index_data.get('name', '').lower()
                if query_lower in name:
                    person = self.find_by_id(person_id)
                    if person:
                        persons.append(person)
                    continue
                
                # 메타데이터 검색 (실제 파일을 로드해야 함)
                person = self.find_by_id(person_id)
                if person and person.metadata:
                    metadata_str = str(person.metadata).lower()
                    if query_lower in metadata_str:
                        persons.append(person)
            
            logger.debug(f"Search '{query}' found {len(persons)} persons")
            return persons
            
        except Exception as e:
            logger.error(f"Error searching persons with query '{query}': {str(e)}")
            return []
    
    def _serialize_person(self, person: Person) -> Dict[str, Any]:
        """인물 객체를 JSON 직렬화 가능한 딕셔너리로 변환"""
        data = {
            'person_id': person.person_id,
            'name': person.name,
            'created_at': person.created_at.isoformat() if person.created_at else None,
            'updated_at': person.updated_at.isoformat() if person.updated_at else None,
            'metadata': person.metadata,
            'face_embeddings': []
        }
        
        # 얼굴 임베딩 직렬화
        for embedding in person.face_embeddings:
            if hasattr(embedding, 'tolist'):
                data['face_embeddings'].append(embedding.tolist())
            else:
                data['face_embeddings'].append(list(embedding))
        
        return data
    
    def _deserialize_person(self, data: Dict[str, Any]) -> Person:
        """JSON 데이터를 인물 객체로 변환"""
        from datetime import datetime
        import numpy as np
        
        # 날짜 복원
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])
        
        # 얼굴 임베딩 복원
        face_embeddings = []
        for embedding_data in data.get('face_embeddings', []):
            embedding = np.array(embedding_data)
            face_embeddings.append(embedding)
        
        return Person(
            person_id=data['person_id'],
            name=data['name'],
            face_embeddings=face_embeddings,
            metadata=data.get('metadata', {}),
            created_at=created_at,
            updated_at=updated_at
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
