#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Matching Service.

얼굴 매칭 및 비교를 위한 도메인 서비스입니다.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from common.logging import get_logger
from ..entities.face import Face
from ..entities.person import Person
from ..value_objects.face_embedding import FaceEmbedding
from ..value_objects.confidence_score import ConfidenceScore

logger = get_logger(__name__)


class FaceMatchingService:
    """얼굴 매칭 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        서비스 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        self.high_confidence_threshold = self.config.get('high_confidence_threshold', 0.8)
        self.max_candidates = self.config.get('max_candidates', 5)
        
        logger.info(f"FaceMatchingService initialized with config: {self.config}")
    
    def find_best_match(self, query_face: Face, candidate_persons: List[Person]) -> Optional[Tuple[Person, float]]:
        """
        쿼리 얼굴에 가장 유사한 인물을 찾습니다.
        
        Args:
            query_face: 검색할 얼굴
            candidate_persons: 후보 인물 리스트
            
        Returns:
            Optional[Tuple[Person, float]]: (매칭된 인물, 유사도 점수) 또는 None
        """
        if query_face.embedding is None:
            logger.warning("쿼리 얼굴에 임베딩이 없습니다")
            return None
        
        if not candidate_persons:
            logger.debug("후보 인물이 없습니다")
            return None
        
        logger.debug(f"최적 매칭 검색 시작 - 후보 수: {len(candidate_persons)}")
        
        best_person = None
        best_similarity = 0.0
        
        try:
            for person in candidate_persons:
                similarity = self._calculate_person_similarity(query_face, person)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_person = person
            
            if best_person:
                logger.info(f"최적 매칭 발견 - 인물: {best_person.name}, 유사도: {best_similarity:.3f}")
                return best_person, best_similarity
            else:
                logger.debug("임계값을 만족하는 매칭을 찾지 못했습니다")
                return None
                
        except Exception as e:
            logger.error(f"최적 매칭 검색 중 오류: {str(e)}")
            return None
    
    def find_top_matches(self, query_face: Face, candidate_persons: List[Person], 
                        top_k: Optional[int] = None) -> List[Tuple[Person, float]]:
        """
        쿼리 얼굴에 유사한 상위 K명의 인물을 찾습니다.
        
        Args:
            query_face: 검색할 얼굴
            candidate_persons: 후보 인물 리스트
            top_k: 반환할 상위 결과 수 (None이면 max_candidates 사용)
            
        Returns:
            List[Tuple[Person, float]]: (인물, 유사도 점수) 리스트 (유사도 순 정렬)
        """
        if query_face.embedding is None:
            logger.warning("쿼리 얼굴에 임베딩이 없습니다")
            return []
        
        if not candidate_persons:
            return []
        
        if top_k is None:
            top_k = self.max_candidates
        
        logger.debug(f"상위 {top_k}개 매칭 검색 시작 - 후보 수: {len(candidate_persons)}")
        
        try:
            # 모든 후보와의 유사도 계산
            similarities = []
            for person in candidate_persons:
                similarity = self._calculate_person_similarity(query_face, person)
                if similarity >= self.similarity_threshold:
                    similarities.append((person, similarity))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 K개 반환
            top_matches = similarities[:top_k]
            
            logger.info(f"상위 매칭 검색 완료 - 발견된 매칭 수: {len(top_matches)}")
            
            return top_matches
            
        except Exception as e:
            logger.error(f"상위 매칭 검색 중 오류: {str(e)}")
            return []
    
    def compare_faces(self, face1: Face, face2: Face) -> Tuple[bool, float, str]:
        """
        두 얼굴을 비교합니다.
        
        Args:
            face1: 첫 번째 얼굴
            face2: 두 번째 얼굴
            
        Returns:
            Tuple[bool, float, str]: (동일 인물 여부, 유사도 점수, 신뢰도 레벨)
        """
        if face1.embedding is None or face2.embedding is None:
            logger.warning("얼굴 중 하나에 임베딩이 없습니다")
            return False, 0.0, "low"
        
        logger.debug(f"얼굴 비교 시작 - Face1: {face1.face_id}, Face2: {face2.face_id}")
        
        try:
            # 유사도 계산
            similarity = face1.embedding.cosine_similarity(face2.embedding)
            
            # 동일 인물 여부 판단
            is_same_person = similarity >= self.similarity_threshold
            
            # 신뢰도 레벨 결정
            if similarity >= self.high_confidence_threshold:
                confidence_level = "high"
            elif similarity >= self.similarity_threshold:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            logger.debug(f"얼굴 비교 완료 - 유사도: {similarity:.3f}, 동일인물: {is_same_person}")
            
            return is_same_person, similarity, confidence_level
            
        except Exception as e:
            logger.error(f"얼굴 비교 중 오류: {str(e)}")
            return False, 0.0, "low"
    
    def batch_compare(self, query_faces: List[Face], reference_faces: List[Face]) -> List[List[Tuple[bool, float]]]:
        """
        여러 얼굴들을 배치로 비교합니다.
        
        Args:
            query_faces: 쿼리 얼굴 리스트
            reference_faces: 참조 얼굴 리스트
            
        Returns:
            List[List[Tuple[bool, float]]]: 각 쿼리 얼굴에 대한 모든 참조 얼굴과의 비교 결과
        """
        if not query_faces or not reference_faces:
            return []
        
        logger.debug(f"배치 비교 시작 - 쿼리: {len(query_faces)}, 참조: {len(reference_faces)}")
        
        results = []
        
        for query_face in query_faces:
            query_results = []
            for reference_face in reference_faces:
                try:
                    is_same, similarity, _ = self.compare_faces(query_face, reference_face)
                    query_results.append((is_same, similarity))
                except Exception as e:
                    logger.warning(f"배치 비교 중 오류: {str(e)}")
                    query_results.append((False, 0.0))
            
            results.append(query_results)
        
        logger.info(f"배치 비교 완료 - 총 {len(query_faces) * len(reference_faces)}개 비교")
        
        return results
    
    def group_similar_faces(self, faces: List[Face], similarity_threshold: Optional[float] = None) -> List[List[Face]]:
        """
        유사한 얼굴들을 그룹화합니다.
        
        Args:
            faces: 그룹화할 얼굴 리스트
            similarity_threshold: 그룹화 임계값 (None이면 기본값 사용)
            
        Returns:
            List[List[Face]]: 얼굴 그룹 리스트
        """
        if not faces:
            return []
        
        threshold = similarity_threshold or self.similarity_threshold
        logger.debug(f"얼굴 그룹화 시작 - 얼굴 수: {len(faces)}, 임계값: {threshold}")
        
        # 임베딩이 없는 얼굴 제외
        valid_faces = [face for face in faces if face.embedding is not None]
        
        if not valid_faces:
            logger.warning("유효한 임베딩을 가진 얼굴이 없습니다")
            return []
        
        groups = []
        processed = set()
        
        try:
            for i, face in enumerate(valid_faces):
                if i in processed:
                    continue
                
                # 새 그룹 시작
                current_group = [face]
                processed.add(i)
                
                # 나머지 얼굴들과 비교
                for j, other_face in enumerate(valid_faces[i+1:], i+1):
                    if j in processed:
                        continue
                    
                    similarity = face.embedding.cosine_similarity(other_face.embedding)
                    if similarity >= threshold:
                        current_group.append(other_face)
                        processed.add(j)
                
                groups.append(current_group)
            
            logger.info(f"얼굴 그룹화 완료 - 그룹 수: {len(groups)}")
            
            return groups
            
        except Exception as e:
            logger.error(f"얼굴 그룹화 중 오류: {str(e)}")
            return []
    
    def _calculate_person_similarity(self, query_face: Face, person: Person) -> float:
        """쿼리 얼굴과 인물 간의 유사도 계산"""
        if query_face.embedding is None:
            return 0.0
        
        # 인물의 평균 임베딩 계산
        avg_embedding = person.get_average_embedding()
        if avg_embedding is None:
            return 0.0
        
        # 코사인 유사도 계산
        return query_face.embedding.cosine_similarity(avg_embedding)
    
    def calculate_confidence_score(self, similarity: float) -> ConfidenceScore:
        """유사도를 기반으로 신뢰도 점수 계산"""
        # 유사도를 0-1 범위의 신뢰도로 변환
        confidence_value = min(max(similarity, 0.0), 1.0)
        return ConfidenceScore(value=confidence_value)
    
    def get_matching_statistics(self, matches: List[Tuple[Person, float]]) -> Dict[str, Any]:
        """매칭 결과 통계 계산"""
        if not matches:
            return {
                "total_matches": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "high_confidence_count": 0
            }
        
        similarities = [similarity for _, similarity in matches]
        
        return {
            "total_matches": len(matches),
            "avg_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "high_confidence_count": sum(1 for s in similarities if s >= self.high_confidence_threshold)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """서비스 통계 정보 반환"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "max_candidates": self.max_candidates,
            "config": self.config
        } 