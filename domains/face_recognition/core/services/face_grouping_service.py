#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 그룹핑 서비스.

같은 사람의 얼굴들을 자동으로 그룹핑하는 서비스입니다.
"""

import os
import json
import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..entities.face import Face
from ..value_objects.face_embedding import FaceEmbedding
from ..value_objects.confidence_score import ConfidenceScore

logger = logging.getLogger(__name__)


class FaceGroup:
    """얼굴 그룹 클래스"""
    
    def __init__(self, group_id: str = None):
        self.group_id = group_id or str(uuid.uuid4())
        self.faces: List[Face] = []
        self.representative_face: Optional[Face] = None
        self.average_embedding: Optional[np.ndarray] = None
        self.group_name: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def add_face(self, face: Face):
        """그룹에 얼굴 추가"""
        self.faces.append(face)
        self.updated_at = time.time()
        self._update_representative_face()
        self._update_average_embedding()
    
    def _update_representative_face(self):
        """대표 얼굴 업데이트 (가장 높은 품질)"""
        if not self.faces:
            self.representative_face = None
            return
        
        # 품질 점수가 가장 높은 얼굴을 대표로 선택
        best_face = max(self.faces, key=lambda f: f.quality_score)
        self.representative_face = best_face
    
    def _update_average_embedding(self):
        """평균 임베딩 업데이트"""
        if not self.faces:
            self.average_embedding = None
            return
        
        embeddings = [face.embedding.values for face in self.faces]
        self.average_embedding = np.mean(embeddings, axis=0)
    
    def get_similarity_to_face(self, face: Face) -> float:
        """새로운 얼굴과의 유사도 계산"""
        if self.average_embedding is None:
            return 0.0
        
        # 코사인 유사도 계산
        embedding1 = self.average_embedding
        embedding2 = face.embedding.values
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def set_group_name(self, name: str):
        """그룹 이름 설정"""
        self.group_name = name
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'group_id': self.group_id,
            'group_name': self.group_name,
            'face_count': len(self.faces),
            'face_ids': [face.face_id for face in self.faces],
            'representative_face_id': self.representative_face.face_id if self.representative_face else None,
            'average_embedding': self.average_embedding.tolist() if self.average_embedding is not None else None,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class FaceGroupingService:
    """얼굴 그룹핑 서비스"""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self.groups: List[FaceGroup] = []
        self.groups_file = Path("data/storage/face_groups.json")
        self._load_groups()
    
    def _load_groups(self):
        """저장된 그룹 로드"""
        try:
            if self.groups_file.exists():
                with open(self.groups_file, 'r', encoding='utf-8') as f:
                    groups_data = json.load(f)
                
                for group_data in groups_data:
                    group = FaceGroup(group_data['group_id'])
                    group.group_name = group_data.get('group_name')
                    group.created_at = group_data.get('created_at', time.time())
                    group.updated_at = group_data.get('updated_at', time.time())
                    
                    if group_data.get('average_embedding'):
                        group.average_embedding = np.array(group_data['average_embedding'])
                    
                    self.groups.append(group)
                
                logger.info(f"Loaded {len(self.groups)} face groups")
        except Exception as e:
            logger.error(f"Error loading face groups: {str(e)}")
            self.groups = []
    
    def _save_groups(self):
        """그룹 저장"""
        try:
            self.groups_file.parent.mkdir(parents=True, exist_ok=True)
            
            groups_data = [group.to_dict() for group in self.groups]
            
            with open(self.groups_file, 'w', encoding='utf-8') as f:
                json.dump(groups_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.groups)} face groups")
        except Exception as e:
            logger.error(f"Error saving face groups: {str(e)}")
    
    def process_face(self, face: Face) -> str:
        """새로운 얼굴을 처리하여 적절한 그룹에 배정"""
        
        # 기존 그룹과 유사도 확인
        best_group = None
        best_similarity = 0.0
        
        for group in self.groups:
            similarity = group.get_similarity_to_face(face)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_group = group
        
        if best_group:
            # 기존 그룹에 추가
            best_group.add_face(face)
            logger.info(f"Face {face.face_id} added to existing group {best_group.group_id} (similarity: {best_similarity:.3f})")
            group_id = best_group.group_id
        else:
            # 새 그룹 생성
            new_group = FaceGroup()
            new_group.add_face(face)
            self.groups.append(new_group)
            logger.info(f"Created new group {new_group.group_id} for face {face.face_id}")
            group_id = new_group.group_id
        
        self._save_groups()
        return group_id
    
    def get_ungrouped_faces(self) -> List[FaceGroup]:
        """이름이 없는 그룹들 반환"""
        return [group for group in self.groups if group.group_name is None]
    
    def get_all_groups(self) -> List[FaceGroup]:
        """모든 그룹 반환"""
        return self.groups.copy()
    
    def set_group_name(self, group_id: str, name: str) -> bool:
        """그룹에 이름 설정"""
        for group in self.groups:
            if group.group_id == group_id:
                group.set_group_name(name)
                self._save_groups()
                logger.info(f"Group {group_id} named as '{name}'")
                return True
        
        logger.warning(f"Group {group_id} not found")
        return False
    
    def merge_groups(self, group_ids: List[str], new_name: str = None) -> bool:
        """여러 그룹을 병합"""
        groups_to_merge = [g for g in self.groups if g.group_id in group_ids]
        
        if len(groups_to_merge) < 2:
            logger.warning("Need at least 2 groups to merge")
            return False
        
        # 첫 번째 그룹을 기준으로 나머지 병합
        main_group = groups_to_merge[0]
        
        for group in groups_to_merge[1:]:
            for face in group.faces:
                main_group.add_face(face)
            self.groups.remove(group)
        
        if new_name:
            main_group.set_group_name(new_name)
        
        self._save_groups()
        logger.info(f"Merged {len(groups_to_merge)} groups into {main_group.group_id}")
        return True
    
    def get_group_by_id(self, group_id: str) -> Optional[FaceGroup]:
        """ID로 그룹 찾기"""
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None
    
    def get_statistics(self) -> Dict:
        """그룹핑 통계"""
        total_groups = len(self.groups)
        named_groups = len([g for g in self.groups if g.group_name])
        total_faces = sum(len(g.faces) for g in self.groups)
        
        return {
            'total_groups': total_groups,
            'named_groups': named_groups,
            'unnamed_groups': total_groups - named_groups,
            'total_faces': total_faces,
            'avg_faces_per_group': total_faces / total_groups if total_groups > 0 else 0
        } 