#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Storage 관리 시스템

이 스크립트는 processed/final에서 data/storage로 데이터를 이동시키는 역할을 합니다.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from common.config import load_config
from domains.face_recognition.infrastructure.storage.file_storage import FileStorage

class StorageManager:
    """Storage 관리 시스템"""
    
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.initialize_storage()
        self.setup_directories()
        
    def setup_logging(self):
        """로깅 설정"""
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """설정 로드"""
        self.config = load_config('config/face_recognition_api.yaml')
        
    def initialize_storage(self):
        """저장소 초기화"""
        storage_config = self.config.get('storage', {})
        self.storage = FileStorage(storage_config)
        
    def setup_directories(self):
        """필요한 디렉토리 생성"""
        self.final_dir = Path("data/domains/face_recognition/processed/final")
        self.storage_dir = Path("domains/face_recognition/data/storage")
        self.persons_dir = self.storage_dir / "persons"
        self.faces_dir = self.storage_dir / "faces"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.archived_dir = Path("data/domains/face_recognition/processed/registered")
        
        # 디렉토리 생성
        for dir_path in [self.persons_dir, self.faces_dir, self.embeddings_dir, self.archived_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def process_final_files(self) -> bool:
        """final 파일들을 storage로 처리"""
        try:
            # final 파일들 확인
            final_files = list(self.final_dir.glob("*.json"))
            if not final_files:
                self.logger.info("처리할 final 파일이 없습니다.")
                return True
                
            self.logger.info(f"{len(final_files)}개의 final 파일 발견")
            
            # 각 파일 처리
            for final_file in final_files:
                self.process_single_final_file(final_file)
                
            # 인덱스 업데이트
            self.update_indexes()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Final 파일 처리 중 오류: {str(e)}")
            return False
            
    def process_single_final_file(self, final_file: Path):
        """단일 final 파일 처리"""
        try:
            self.logger.info(f"처리 중: {final_file.name}")
            
            # JSON 파일 로드
            with open(final_file, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
                
            # 중복 검사
            existing_person = self.check_existing_person(final_data)
            
            if existing_person:
                # 기존 Person에 Face 추가
                self.add_face_to_existing_person(existing_person, final_data)
            else:
                # 새로운 Person 생성
                self.create_new_person_and_face(final_data)
                
            # 파일 아카이브
            self.archive_final_file(final_file)
            
        except Exception as e:
            self.logger.error(f"파일 {final_file.name} 처리 중 오류: {str(e)}")
            
    def check_existing_person(self, final_data: Dict) -> Optional[Dict]:
        """기존 Person 검사"""
        try:
            person_name = final_data["person_name"]
            
            # 이름으로 검색
            existing_person = self.storage.find_person_by_name(person_name)
            
            if existing_person:
                return existing_person
                
            # 임베딩 유사도로 검색 (선택적)
            embedding = final_data["face_data"]["embedding"]
            similar_faces = self.find_similar_faces(embedding)
            
            if similar_faces:
                # 가장 유사한 얼굴의 Person 반환
                best_match = similar_faces[0]
                person_id = best_match["person_id"]
                return self.storage.get_person(person_id)
                
            return None
            
        except Exception as e:
            self.logger.error(f"기존 Person 검사 중 오류: {str(e)}")
            return None
            
    def find_similar_faces(self, embedding: List[float], threshold: float = 0.8) -> List[Dict]:
        """유사한 얼굴 검색"""
        try:
            # 모든 얼굴 임베딩 로드
            all_faces = self.storage.get_all_faces()
            
            similar_faces = []
            for face_id, face_data in all_faces.items():
                face_embedding = face_data["embedding"]
                similarity = self.calculate_similarity(embedding, face_embedding)
                
                if similarity > threshold:
                    similar_faces.append({
                        "face_id": face_id,
                        "person_id": face_data["person_id"],
                        "similarity": similarity
                    })
                    
            # 유사도 순으로 정렬
            similar_faces.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_faces
            
        except Exception as e:
            self.logger.error(f"유사 얼굴 검색 중 오류: {str(e)}")
            return []
            
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """코사인 유사도 계산"""
        try:
            import numpy as np
            
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 정규화
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # 코사인 유사도
            similarity = np.dot(vec1_norm, vec2_norm)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"유사도 계산 중 오류: {str(e)}")
            return 0.0
            
    def add_face_to_existing_person(self, person: Dict, final_data: Dict):
        """기존 Person에 Face 추가"""
        try:
            person_id = person["id"]
            
            # Face 생성
            face_id = self.create_face_record(person_id, final_data)
            
            # Person 정보 업데이트
            self.update_person_face_count(person_id, face_id)
            
            self.logger.info(f"기존 인물에 얼굴 추가: {person['name']} (Face ID: {face_id})")
            
        except Exception as e:
            self.logger.error(f"기존 Person에 Face 추가 중 오류: {str(e)}")
            
    def create_new_person_and_face(self, final_data: Dict):
        """새로운 Person과 Face 생성"""
        try:
            # Person 생성
            person_id = self.create_person_record(final_data)
            
            # Face 생성
            face_id = self.create_face_record(person_id, final_data)
            
            # Person 정보 업데이트
            self.update_person_face_count(person_id, face_id)
            
            self.logger.info(f"새 인물 등록: {final_data['person_name']} (Person ID: {person_id}, Face ID: {face_id})")
            
        except Exception as e:
            self.logger.error(f"새 Person/Face 생성 중 오류: {str(e)}")
            
    def create_person_record(self, final_data: Dict) -> str:
        """Person 레코드 생성"""
        person_id = f"person_{uuid.uuid4().hex[:8]}"
        
        person_data = {
            "id": person_id,
            "name": final_data["person_name"],
            "face_ids": [],
            "primary_face_id": None,
            
            "statistics": {
                "total_faces": 1,
                "avg_quality_score": final_data["quality_metrics"]["overall_score"],
                "first_registered": final_data["created_at"],
                "last_updated": final_data["created_at"]
            },
            
            "metadata": {
                "registration_methods": [final_data["source_method"]],
                "source_cameras": [final_data["metadata"]["camera_id"]]
            },
            
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 파일 저장
        person_file = self.persons_dir / f"{person_id}.json"
        with open(person_file, 'w', encoding='utf-8') as f:
            json.dump(person_data, f, ensure_ascii=False, indent=2)
            
        return person_id
        
    def create_face_record(self, person_id: str, final_data: Dict) -> str:
        """Face 레코드 생성"""
        face_id = f"face_{uuid.uuid4().hex[:8]}"
        
        face_data = {
            "id": face_id,
            "person_id": person_id,
            "embedding": final_data["face_data"]["embedding"],
            
            "quality_metrics": final_data["quality_metrics"],
            
            "geometric_data": {
                "bbox": final_data["face_data"]["bbox"],
                "landmarks": final_data["face_data"].get("landmarks", []),
                "pose": final_data["face_data"].get("pose", {"yaw": 0, "pitch": 0, "roll": 0})
            },
            
            "source_info": {
                "method": final_data["source_method"],
                "original_path": final_data["original_image_path"],
                "processed_path": final_data["processed_image_path"],
                "camera_id": final_data["metadata"]["camera_id"]
            },
            
            "model_info": final_data["metadata"]["model_versions"],
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 파일 저장
        face_file = self.faces_dir / f"{face_id}.json"
        with open(face_file, 'w', encoding='utf-8') as f:
            json.dump(face_data, f, ensure_ascii=False, indent=2)
            
        # 임베딩 캐시 저장
        embedding_file = self.embeddings_dir / f"{face_id}.npy"
        import numpy as np
        np.save(embedding_file, np.array(face_data["embedding"]))
        
        return face_id
        
    def update_person_face_count(self, person_id: str, face_id: str):
        """Person의 얼굴 수 업데이트"""
        try:
            person_file = self.persons_dir / f"{person_id}.json"
            
            with open(person_file, 'r', encoding='utf-8') as f:
                person_data = json.load(f)
                
            # Face ID 추가
            if face_id not in person_data["face_ids"]:
                person_data["face_ids"].append(face_id)
                
            # Primary face 설정 (첫 번째 얼굴)
            if person_data["primary_face_id"] is None:
                person_data["primary_face_id"] = face_id
                
            # 통계 업데이트
            person_data["statistics"]["total_faces"] = len(person_data["face_ids"])
            person_data["statistics"]["last_updated"] = datetime.now().isoformat()
            person_data["updated_at"] = datetime.now().isoformat()
            
            # 파일 저장
            with open(person_file, 'w', encoding='utf-8') as f:
                json.dump(person_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Person 업데이트 중 오류: {str(e)}")
            
    def archive_final_file(self, final_file: Path):
        """Final 파일 아카이브"""
        try:
            # 아카이브 디렉토리로 이동
            archive_path = self.archived_dir / final_file.name
            shutil.move(str(final_file), str(archive_path))
            
            # 관련 이미지 파일도 이동
            image_file = final_file.with_suffix('.jpg')
            if image_file.exists():
                archive_image_path = self.archived_dir / image_file.name
                shutil.move(str(image_file), str(archive_image_path))
                
        except Exception as e:
            self.logger.error(f"파일 아카이브 중 오류: {str(e)}")
            
    def update_indexes(self):
        """검색 인덱스 업데이트"""
        try:
            # Person 인덱스 생성
            self.create_person_index()
            
            # Face 인덱스 생성
            self.create_face_index()
            
            self.logger.info("인덱스 업데이트 완료")
            
        except Exception as e:
            self.logger.error(f"인덱스 업데이트 중 오류: {str(e)}")
            
    def create_person_index(self):
        """Person 검색 인덱스 생성"""
        try:
            person_files = list(self.persons_dir.glob("*.json"))
            
            index_data = {
                "version": "1.0.0",
                "total_persons": len(person_files),
                "last_updated": datetime.now().isoformat(),
                "name_index": {},
                "id_index": {}
            }
            
            for person_file in person_files:
                with open(person_file, 'r', encoding='utf-8') as f:
                    person_data = json.load(f)
                    
                person_id = person_data["id"]
                person_name = person_data["name"]
                
                # 이름 인덱스
                index_data["name_index"][person_name] = person_id
                
                # ID 인덱스
                index_data["id_index"][person_id] = {
                    "name": person_name,
                    "face_count": len(person_data["face_ids"]),
                    "primary_face_id": person_data["primary_face_id"]
                }
                
            # 인덱스 파일 저장
            index_file = self.storage_dir / "person_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Person 인덱스 생성 중 오류: {str(e)}")
            
    def create_face_index(self):
        """Face 검색 인덱스 생성"""
        try:
            face_files = list(self.faces_dir.glob("*.json"))
            
            index_data = {
                "version": "1.0.0",
                "total_faces": len(face_files),
                "last_updated": datetime.now().isoformat(),
                "person_mapping": {},
                "quality_sorted": []
            }
            
            for face_file in face_files:
                with open(face_file, 'r', encoding='utf-8') as f:
                    face_data = json.load(f)
                    
                face_id = face_data["id"]
                person_id = face_data["person_id"]
                quality = face_data["quality_metrics"]["overall_score"]
                
                # Person 매핑
                if person_id not in index_data["person_mapping"]:
                    index_data["person_mapping"][person_id] = []
                index_data["person_mapping"][person_id].append(face_id)
                
                # 품질 정렬용
                index_data["quality_sorted"].append({
                    "face_id": face_id,
                    "quality": quality
                })
                
            # 품질 순으로 정렬
            index_data["quality_sorted"].sort(key=lambda x: x["quality"], reverse=True)
            
            # 인덱스 파일 저장
            index_file = self.storage_dir / "face_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Face 인덱스 생성 중 오류: {str(e)}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Storage 관리 시스템")
    parser.add_argument("--force", action="store_true", help="강제 처리")
    args = parser.parse_args()
    
    manager = StorageManager()
    
    try:
        success = manager.process_final_files()
        if success:
            print("Storage 처리가 성공적으로 완료되었습니다.")
            print("데이터가 domains/face_recognition/data/storage/에 저장되었습니다.")
        else:
            print("Storage 처리 중 오류가 발생했습니다.")
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")

if __name__ == "__main__":
    main() 