#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🧠 스마트 배치 처리기 (자동 그룹핑)

data/temp/uploads/ → 모든 얼굴 검출 → 자동 그룹핑 → 그룹별 이름 지정 → face_staging

예시:
uploads/에 10개 파일 → 총 25개 얼굴 검출 
→ 3개 그룹으로 자동 분류:
   그룹1: 홍길동 (8개 얼굴)
   그룹2: 김철수 (12개 얼굴)  
   그룹3: 이영희 (5개 얼굴)
→ 3번만 이름 입력하면 끝!
"""

import os
import sys
import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.infrastructure.detection_engines.retinaface_detection_engine import RetinaFaceDetectionEngine
from domains.face_recognition.infrastructure.models.arcface_recognizer import ArcFaceRecognizer

# sklearn import 시도
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn이 없어서 간단한 거리 기반 그룹핑을 사용합니다.")

logger = logging.getLogger(__name__)


class SmartBatchProcessor:
    """🧠 스마트 배치 처리기 - 자동 그룹핑 지원"""
    
    def __init__(self):
        """초기화"""
        # 🎯 폴더 설정
        self.upload_dir = Path("data/temp/uploads")
        self.face_staging_dir = Path("data/temp/face_staging")
        
        # 폴더 생성
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.face_staging_dir.mkdir(parents=True, exist_ok=True)
        
        # AI 모델 초기화
        print("🤖 AI 모델 로딩 중...")
        self.face_detector = RetinaFaceDetectionEngine()
        self.face_recognizer = ArcFaceRecognizer()
        
        # 처리 결과 저장
        self.all_faces = []  # 모든 검출된 얼굴 정보
        self.face_groups = []  # 그룹핑 결과
        
        print("✅ 스마트 배치 처리기 준비 완료!")
    
    def run(self):
        """메인 실행"""
        print("\n" + "="*60)
        print("🧠 스마트 배치 처리기 (자동 그룹핑)")
        print("="*60)
        print("📂 업로드 폴더:", self.upload_dir)
        print("🎯 결과 저장소:", self.face_staging_dir)
        print()
        
        # 1단계: 업로드 파일 스캔
        files = self._scan_upload_files()
        if not files:
            print("❌ 처리할 파일이 없습니다.")
            print(f"   📁 {self.upload_dir} 폴더에 이미지나 동영상을 넣어주세요.")
            return
        
        print(f"📋 발견된 파일: {len(files)}개")
        for i, file_path in enumerate(files, 1):
            print(f"  {i}. {file_path.name}")
        
        # 사용자 확인
        if not self._confirm_processing(files):
            print("취소되었습니다.")
            return
        
        try:
            # 2단계: 모든 파일에서 얼굴 검출
            print("\n🔍 1단계: 모든 파일에서 얼굴 검출 중...")
            self._detect_all_faces(files)
            
            if not self.all_faces:
                print("❌ 검출된 얼굴이 없습니다.")
                return
            
            print(f"✅ 총 {len(self.all_faces)}개 얼굴 검출 완료!")
            
            # 3단계: 임베딩 생성 및 자동 그룹핑
            print("\n🧠 2단계: AI 기반 자동 그룹핑 중...")
            self._auto_group_faces()
            
            print(f"✅ {len(self.face_groups)}개 그룹으로 자동 분류 완료!")
            
            # 4단계: 그룹 확인 및 이름 지정
            print("\n👥 3단계: 그룹별 이름 지정...")
            if not self._assign_group_names():
                return
            
            # 5단계: face_staging으로 이동
            print("\n📂 4단계: 공통 허브로 이동...")
            self._move_to_face_staging()
            
            print("\n🎉 스마트 배치 처리 완료!")
            print(f"   📂 결과: {self.face_staging_dir}")
            print("   🎯 이제 기존 시스템과 동일한 분기 처리가 가능합니다!")
            
        except Exception as e:
            logger.error(f"스마트 배치 처리 중 오류: {str(e)}")
            print(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
    
    def _scan_upload_files(self) -> List[Path]:
        """업로드 폴더 스캔"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
        files = []
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        return sorted(files)
    
    def _confirm_processing(self, files: List[Path]) -> bool:
        """처리 확인"""
        print(f"\n📋 {len(files)}개 파일을 스마트 배치 처리하시겠습니까?")
        print("   🧠 AI가 자동으로 같은 사람끼리 그룹핑합니다")
        print("   🏷️ 그룹별로 한 번만 이름을 입력하면 됩니다")
        
        while True:
            choice = input("\n계속하시겠습니까? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("y 또는 n을 입력해주세요.")
    
    def _detect_all_faces(self, files: List[Path]):
        """모든 파일에서 얼굴 검출"""
        total_files = len(files)
        
        for i, file_path in enumerate(files, 1):
            print(f"  [{i}/{total_files}] {file_path.name} 처리 중...")
            
            try:
                if file_path.suffix.lower() in {'.mp4', '.avi', '.mov'}:
                    self._detect_faces_in_video(file_path)
                else:
                    self._detect_faces_in_image(file_path)
                    
            except Exception as e:
                logger.error(f"파일 처리 실패 {file_path}: {str(e)}")
                print(f"    ❌ 처리 실패: {str(e)}")
    
    def _detect_faces_in_image(self, image_path: Path):
        """이미지에서 얼굴 검출"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"    ❌ 이미지 로드 실패: {image_path}")
            return
        
        # 얼굴 검출
        faces = self.face_detector.detect_faces(image)
        
        for j, face in enumerate(faces):
            face_info = {
                'source_path': str(image_path),
                'source_type': 'image',
                'face_index': j,
                'face_data': face,
                'image': image,
                'timestamp': time.time()
            }
            self.all_faces.append(face_info)
        
        print(f"    ✅ {len(faces)}개 얼굴 검출")
    
    def _detect_faces_in_video(self, video_path: Path):
        """동영상에서 얼굴 검출 (샘플링)"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    ❌ 동영상 로드 실패: {video_path}")
            return
        
        # 동영상 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 5초마다 샘플링 (너무 많으면 느려짐)
        sample_interval = max(1, int(fps * 5))
        frame_count = 0
        detected_faces = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 샘플링 간격에 맞는 프레임만 처리
            if frame_count % sample_interval == 0:
                faces = self.face_detector.detect_faces(frame)
                
                for j, face in enumerate(faces):
                    face_info = {
                        'source_path': str(video_path),
                        'source_type': 'video',
                        'face_index': f"{frame_count}_{j}",
                        'frame_number': frame_count,
                        'timestamp_video': frame_count / fps,
                        'face_data': face,
                        'image': frame.copy(),
                        'timestamp': time.time()
                    }
                    self.all_faces.append(face_info)
                    detected_faces += 1
            
            frame_count += 1
        
        cap.release()
        print(f"    ✅ {detected_faces}개 얼굴 검출 (총 {total_frames}프레임 중 샘플링)")
    
    def _auto_group_faces(self):
        """AI 기반 자동 그룹핑"""
        if len(self.all_faces) < 2:
            # 얼굴이 1개면 그룹핑 불가
            self.face_groups = [{'faces': self.all_faces, 'representative': self.all_faces[0]}]
            return
        
        print("  🧠 얼굴 임베딩 생성 중...")
        embeddings = []
        valid_faces = []
        
        # 모든 얼굴의 임베딩 생성
        for i, face_info in enumerate(self.all_faces):
            try:
                face_data = face_info['face_data']
                image = face_info['image']
                
                # 얼굴 크롭
                x, y, w, h = face_data.bbox.to_list()
                face_crop = image[y:y+h, x:x+w]
                
                if face_crop.size > 0:
                    # 임베딩 생성
                    embedding = self.face_recognizer.extract_embedding(face_crop)
                    embeddings.append(embedding)
                    valid_faces.append(face_info)
                    
            except Exception as e:
                logger.warning(f"임베딩 생성 실패 (face {i}): {str(e)}")
        
        if len(embeddings) < 2:
            self.face_groups = [{'faces': valid_faces, 'representative': valid_faces[0]}]
            return
        
        print(f"  📊 {len(embeddings)}개 임베딩 생성 완료")
        print("  🔗 유사도 기반 클러스터링 중...")
        
        # 그룹핑 수행
        if SKLEARN_AVAILABLE:
            self._group_with_sklearn(embeddings, valid_faces)
        else:
            self._group_with_simple_clustering(embeddings, valid_faces)
        
        # 크기 순으로 정렬 (큰 그룹부터)
        self.face_groups.sort(key=lambda g: g['size'], reverse=True)
        
        print(f"  ✅ {len(self.face_groups)}개 그룹으로 분류 완료!")
        for i, group in enumerate(self.face_groups):
            print(f"    그룹 {i+1}: {group['size']}개 얼굴")
    
    def _group_with_sklearn(self, embeddings: List, valid_faces: List):
        """sklearn을 사용한 고급 클러스터링"""
        embeddings_array = np.array(embeddings)
        
        # DBSCAN 클러스터링으로 그룹핑 (거리 임계값 0.4 = 60% 유사도)
        clustering = DBSCAN(eps=0.4, min_samples=1, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings_array)
        
        # 그룹별로 얼굴들 분류
        groups_dict = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            groups_dict[label].append((valid_faces[i], embeddings[i]))
        
        # 그룹 정보 구성
        self.face_groups = []
        for group_id, faces_with_embeddings in groups_dict.items():
            faces = [item[0] for item in faces_with_embeddings]
            
            # 대표 얼굴 선택 (가장 높은 품질 점수)
            best_face = max(faces, key=lambda f: f['face_data'].confidence.value)
            
            group_info = {
                'group_id': group_id,
                'faces': faces,
                'representative': best_face,
                'size': len(faces)
            }
            self.face_groups.append(group_info)
    
    def _group_with_simple_clustering(self, embeddings: List, valid_faces: List):
        """간단한 거리 기반 클러스터링"""
        groups = []
        used_indices = set()
        threshold = 0.4  # 코사인 거리 임계값
        
        for i, embedding in enumerate(embeddings):
            if i in used_indices:
                continue
            
            # 새 그룹 시작
            group_faces = [valid_faces[i]]
            used_indices.add(i)
            
            # 유사한 얼굴들 찾기
            for j, other_embedding in enumerate(embeddings):
                if j in used_indices:
                    continue
                
                # 코사인 유사도 계산
                cosine_sim = np.dot(embedding, other_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                )
                cosine_distance = 1 - cosine_sim
                
                if cosine_distance < threshold:
                    group_faces.append(valid_faces[j])
                    used_indices.add(j)
            
            # 대표 얼굴 선택
            best_face = max(group_faces, key=lambda f: f['face_data'].confidence.value)
            
            group_info = {
                'group_id': len(groups),
                'faces': group_faces,
                'representative': best_face,
                'size': len(group_faces)
            }
            groups.append(group_info)
        
        self.face_groups = groups
    
    def _assign_group_names(self) -> bool:
        """그룹별 이름 지정"""
        print(f"\n👥 {len(self.face_groups)}개 그룹에 이름을 지정해주세요:")
        print("   'skip' 입력시 건너뛰기, 'quit' 입력시 중단")
        
        for i, group in enumerate(self.face_groups):
            print(f"\n--- 그룹 {i+1}/{len(self.face_groups)} ({group['size']}개 얼굴) ---")
            
            # 대표 얼굴 표시
            self._show_group_representative(group, i+1)
            
            # 전체 그룹 얼굴들 표시 여부 확인
            if group['size'] > 1:
                show_all = input(f"그룹의 모든 얼굴을 보시겠습니까? (y/n, 기본값: n): ").strip().lower()
                if show_all in ['y', 'yes']:
                    self._show_group_faces(group, i+1)
            
            # 이름 입력
            while True:
                group_name = input(f"그룹 {i+1} 이름을 입력하세요: ").strip()
                
                if group_name.lower() == 'quit':
                    print("처리를 중단합니다.")
                    return False
                elif group_name.lower() == 'skip':
                    print(f"그룹 {i+1}을 건너뜁니다.")
                    group['name'] = None
                    break
                elif group_name:
                    group['name'] = group_name
                    print(f"✅ '{group_name}' 이름이 설정되었습니다!")
                    break
                else:
                    print("이름을 입력해주세요. (skip으로 건너뛰기 가능)")
        
        return True
    
    def _show_group_representative(self, group: Dict, group_num: int):
        """그룹 대표 얼굴 표시"""
        rep_face = group['representative']
        face_data = rep_face['face_data']
        image = rep_face['image']
        
        # 얼굴 크롭
        x, y, w, h = face_data.bbox.to_list()
        face_crop = image[y:y+h, x:x+w]
        
        if face_crop.size > 0:
            # 리사이즈 및 정보 표시
            display_size = (200, 200)
            face_display = cv2.resize(face_crop, display_size)
            
            # 정보 오버레이
            cv2.putText(face_display, f"Group {group_num}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(face_display, f"{group['size']} faces", (5, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(face_display, f"Conf: {face_data.confidence.value:.2f}", (5, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            window_name = f"Group {group_num} Representative"
            cv2.imshow(window_name, face_display)
            
            print(f"📷 그룹 {group_num} 대표 얼굴이 표시됩니다. 아무 키나 누르면 계속...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
    
    def _show_group_faces(self, group: Dict, group_num: int):
        """그룹의 모든 얼굴 표시"""
        faces = group['faces']
        faces_per_row = 4
        rows = (len(faces) + faces_per_row - 1) // faces_per_row
        
        # 얼굴 이미지들 준비
        face_images = []
        for face_info in faces:
            face_data = face_info['face_data']
            image = face_info['image']
            
            x, y, w, h = face_data.bbox.to_list()
            face_crop = image[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (120, 120))
                
                # 정보 표시
                cv2.putText(face_resized, f"{face_data.confidence.value:.2f}", (5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                face_images.append(face_resized)
        
        # 그리드 생성
        if face_images:
            grid_rows = []
            for row in range(rows):
                start_idx = row * faces_per_row
                end_idx = min(start_idx + faces_per_row, len(face_images))
                row_images = face_images[start_idx:end_idx]
                
                # 빈 공간 채우기
                while len(row_images) < faces_per_row:
                    row_images.append(np.zeros((120, 120, 3), dtype=np.uint8))
                
                row_img = np.hstack(row_images)
                grid_rows.append(row_img)
            
            grid_img = np.vstack(grid_rows)
            
            window_name = f"Group {group_num} All Faces ({len(faces)} faces)"
            cv2.imshow(window_name, grid_img)
            
            print(f"📷 그룹 {group_num}의 모든 얼굴이 표시됩니다. 아무 키나 누르면 계속...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
    
    def _move_to_face_staging(self):
        """face_staging으로 이동"""
        moved_count = 0
        
        for group in self.face_groups:
            group_name = group.get('name')
            if not group_name:
                print(f"⚠️ 이름이 없는 그룹 {group['group_id']} (얼굴 {group['size']}개)를 건너뜁니다.")
                continue
            
            # 그룹의 모든 얼굴을 face_staging으로 이동
            for face_info in group['faces']:
                self._save_face_to_staging(face_info, group_name)
                moved_count += 1
        
        print(f"✅ 총 {moved_count}개 얼굴이 face_staging으로 이동되었습니다.")
    
    def _save_face_to_staging(self, face_info: Dict, person_name: str):
        """개별 얼굴을 face_staging에 저장"""
        timestamp = int(time.time())
        face_index = face_info['face_index']
        
        face_data = face_info['face_data']
        image = face_info['image']
        
        # 전체 이미지 저장
        frame_filename = f"{person_name}_{timestamp}_frame_{face_index}.jpg"
        frame_path = self.face_staging_dir / frame_filename
        cv2.imwrite(str(frame_path), image)
        
        # 얼굴 크롭 저장
        x, y, w, h = face_data.bbox.to_list()
        face_crop = image[y:y+h, x:x+w]
        face_filename = f"{person_name}_{timestamp}_face_{face_index}.jpg"
        face_path = self.face_staging_dir / face_filename
        cv2.imwrite(str(face_path), face_crop)
        
        # 메타데이터 저장
        metadata = {
            'person_name': person_name,
            'timestamp': timestamp,
            'face_index': face_index,
            'bbox': face_data.bbox.to_list(),
            'confidence': face_data.confidence.value,
            'source_path': face_info['source_path'],
            'source_type': face_info['source_type'],
            'processing_type': 'smart_batch',
            'frame_path': str(frame_path),
            'face_path': str(face_path),
            'created_at': datetime.now().isoformat()
        }
        
        # 동영상인 경우 추가 정보
        if face_info['source_type'] == 'video':
            metadata['frame_number'] = face_info.get('frame_number')
            metadata['timestamp_video'] = face_info.get('timestamp_video')
        
        metadata_filename = f"{person_name}_{timestamp}_meta_{face_index}.json"
        metadata_path = self.face_staging_dir / metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    """메인 함수"""
    # 로깅 설정
    setup_logging()
    
    try:
        processor = SmartBatchProcessor()
        processor.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"스마트 배치 처리기 오류: {str(e)}")
        print(f"\n❌ 오류가 발생했습니다: {str(e)}")
        raise


if __name__ == "__main__":
    main() 