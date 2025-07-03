#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 유사도 기반 그룹핑 스크립트

- detected_faces에 저장된 얼굴들을 유사도 기반으로 그룹핑
- 유사한 얼굴들을 같은 그룹으로 분류하여 staging/grouped에 저장
- 얼굴 임베딩 추출 및 유사도 계산
"""

import cv2
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict
import shutil

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 범용 네이밍 시스템 import
try:
    from shared.vision_core.naming import UniversalNamingSystem, UniversalMetadata
except ImportError:
    # 상대 경로로 import 시도
    sys.path.insert(0, str(project_root / "shared"))
    from vision_core.naming import UniversalNamingSystem, UniversalMetadata

# 경로 설정
DETECTED_DIR = Path("data/domains/face_recognition/detected_faces")
GROUPED_DIR = Path("data/domains/face_recognition/staging/grouped")
GROUPED_DIR.mkdir(parents=True, exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def load_face_recognizer():
    """얼굴 인식기 로딩"""
    try:
        from domains.face_recognition.infrastructure.models.mobilefacenet_recognizer import MobileFaceNetRecognizer
        model_path = "models/weights/face_recognition_mobilefacenet_20250628.onnx"
        if Path(model_path).exists():
            logger.info("[MobileFaceNet] 얼굴 인식기 사용")
            recognizer = MobileFaceNetRecognizer(model_path=model_path, device="auto")
            return recognizer
        else:
            logger.warning(f"[MobileFaceNet] 모델 파일이 존재하지 않습니다: {model_path}")
    except Exception as e:
        logger.warning(f"[경고] MobileFaceNet 로딩 실패: {e}")
    
    logger.error("[치명적] 사용 가능한 얼굴 인식기가 없습니다.")
    return None

def extract_face_embeddings(recognizer, face_images: List[Tuple[Path, np.ndarray]]) -> List[Tuple[Path, np.ndarray, np.ndarray]]:
    """얼굴 이미지에서 임베딩 추출"""
    embeddings = []
    
    for face_path, face_img in face_images:
        try:
            # 얼굴 이미지 전처리 (크기 조정)
            face_resized = cv2.resize(face_img, (112, 112))
            
            # 임베딩 추출
            face_embedding = recognizer.extract_embedding(face_resized)
            
            if face_embedding is not None:
                # FaceEmbedding에서 vector 추출
                embedding_vector = face_embedding.vector
                embeddings.append((face_path, face_img, embedding_vector))
                logger.debug(f"임베딩 추출 성공: {face_path.name}")
            else:
                logger.warning(f"임베딩 추출 실패: {face_path.name}")
                
        except Exception as e:
            logger.error(f"임베딩 추출 오류 {face_path.name}: {e}")
    
    return embeddings

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """두 임베딩 간의 코사인 유사도 계산"""
    # 정규화
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 코사인 유사도 계산
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(similarity)

def group_similar_faces(embeddings: List[Tuple[Path, np.ndarray, np.ndarray]], 
                       similarity_threshold: float = 0.6) -> List[List[Tuple[Path, np.ndarray, np.ndarray]]]:
    """유사도 기반 얼굴 그룹핑"""
    if not embeddings:
        return []
    
    groups = []
    used_indices = set()
    
    for i, (face_path, face_img, embedding) in enumerate(embeddings):
        if i in used_indices:
            continue
        
        # 새 그룹 시작
        current_group = [(face_path, face_img, embedding)]
        used_indices.add(i)
        
        # 다른 얼굴들과 유사도 비교
        for j, (other_path, other_img, other_embedding) in enumerate(embeddings):
            if j in used_indices or i == j:
                continue
            
            similarity = calculate_similarity(embedding, other_embedding)
            
            if similarity >= similarity_threshold:
                current_group.append((other_path, other_img, other_embedding))
                used_indices.add(j)
                logger.debug(f"그룹 추가: {other_path.name} (유사도: {similarity:.3f})")
        
        if len(current_group) > 1:  # 2개 이상의 얼굴이 있는 그룹만 저장
            groups.append(current_group)
            logger.info(f"그룹 {len(groups)} 생성: {len(current_group)}개 얼굴")
    
    return groups

def save_face_group(group: List[Tuple[Path, np.ndarray, np.ndarray]], group_id: int):
    """얼굴 그룹을 파일로 저장"""
    group_dir = GROUPED_DIR / f"group_{group_id:03d}"
    group_dir.mkdir(exist_ok=True)
    
    # 그룹 메타데이터 생성
    group_meta = {
        "group_id": group_id,
        "face_count": len(group),
        "created_at": get_timestamp(),
        "faces": []
    }
    
    for i, (face_path, face_img, embedding) in enumerate(group):
        # 범용 네이밍으로 그룹 파일명 생성
        face_filename = UniversalNamingSystem.create_group_filename(
            domain='face_recognition',
            group_id=group_id,
            sequence=i + 1
        )
        
        face_filepath = group_dir / face_filename
        cv2.imwrite(str(face_filepath), face_img)
        
        # 메타데이터에 추가
        face_meta = {
            "index": i + 1,
            "original_file": str(face_path),
            "saved_as": face_filename,
            "embedding_shape": embedding.shape
        }
        group_meta["faces"].append(face_meta)
    
    # 그룹 메타데이터 저장
    meta_filepath = group_dir / "group_metadata.json"
    with open(meta_filepath, 'w', encoding='utf-8') as f:
        json.dump(group_meta, f, ensure_ascii=False, indent=2)
    
    logger.info(f"그룹 {group_id} 저장 완료: {group_dir}")

def load_all_face_images() -> List[Tuple[Path, np.ndarray]]:
    """detected_faces에서 모든 얼굴 이미지 로드"""
    face_images = []
    
    for date_dir in DETECTED_DIR.iterdir():
        if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
            for img_file in date_dir.glob("*.jpg"):
                try:
                    face_img = cv2.imread(str(img_file))
                    if face_img is not None:
                        face_images.append((img_file, face_img))
                except Exception as e:
                    logger.warning(f"이미지 로드 실패: {img_file} - {e}")
    
    logger.info(f"총 {len(face_images)}개의 얼굴 이미지 로드")
    return face_images

def main():
    # 얼굴 인식기 로딩
    recognizer = load_face_recognizer()
    if recognizer is None:
        logger.error("[치명적] 얼굴 인식기를 사용할 수 없습니다.")
        sys.exit(1)
    
    # 모든 얼굴 이미지 로드
    face_images = load_all_face_images()
    if not face_images:
        logger.warning("처리할 얼굴 이미지가 없습니다.")
        return
    
    # 얼굴 임베딩 추출
    logger.info("얼굴 임베딩 추출 시작...")
    embeddings = extract_face_embeddings(recognizer, face_images)
    
    if not embeddings:
        logger.error("임베딩을 추출할 수 없습니다.")
        return
    
    logger.info(f"임베딩 추출 완료: {len(embeddings)}개")
    
    # 유사도 기반 그룹핑
    logger.info("얼굴 그룹핑 시작...")
    groups = group_similar_faces(embeddings, similarity_threshold=0.6)
    
    if not groups:
        logger.info("유사한 얼굴 그룹을 찾을 수 없습니다.")
        return
    
    # 그룹 저장
    logger.info(f"총 {len(groups)}개 그룹 저장 시작...")
    for i, group in enumerate(groups):
        save_face_group(group, i + 1)
    
    logger.info(f"얼굴 그룹핑 완료! {len(groups)}개 그룹이 {GROUPED_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    main() 