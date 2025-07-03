#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
배치 얼굴 처리 시스템 (업로드 파일용)

📋 업로드 데이터 흐름 (사용자 제안):
   사진/동영상 업로드 → data/domains/face_recognition/raw_input/uploads → 얼굴 검출 → 사용자 선택/이름지정 → data/domains/face_recognition/staging
                                                                                    ↓
                                                                            🎯 기존 시스템과 동일:
                                                                            1️⃣ 즉시 등록 (임베딩 → storage)
                                                                            2️⃣ 훈련용 수집 (품질평가 → datasets)
"""

import cv2
import numpy as np
import time
import sys
import os
import uuid
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from shared.vision_core.quality.face_quality_assessor import CustomFaceQualityAssessor
from common.logging import setup_logging

logger = logging.getLogger(__name__)

class BatchFaceProcessor:
    """배치 얼굴 처리 시스템"""
    
    def __init__(self):
        """초기화"""
        self.detection_service = FaceDetectionService()
        self.recognition_service = FaceRecognitionService()
        self.quality_assessor = CustomFaceQualityAssessor()
        
        # 🎯 새로운 경로 구조 적용
        self.upload_dir = Path("data/domains/face_recognition/raw_input/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 🎯 face_staging 폴더 (공통 허브) - 새로운 경로
        self.face_staging_dir = Path("data/domains/face_recognition/staging")
        self.face_staging_dir.mkdir(parents=True, exist_ok=True)
        
        # 지원 파일 형식
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        
    def run_interactive_mode(self):
        """대화형 모드 실행"""
        print("📁 배치 얼굴 처리 시스템")
        print("=" * 50)
        print("📋 지원 기능:")
        print("  1. 이미지 파일에서 얼굴 검출")
        print("  2. 동영상 파일에서 얼굴 검출")
        print("  3. 업로드 폴더 일괄 처리")
        print("  0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-3): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    self._process_image_files()
                elif choice == '2':
                    self._process_video_files()
                elif choice == '3':
                    self._process_upload_folder()
                else:
                    print("❌ 잘못된 선택입니다.")
                    
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
    
    def _process_image_files(self):
        """이미지 파일 처리"""
        print("\n📷 이미지 파일 처리")
        
        # 파일 경로 입력
        file_path = input("이미지 파일 경로를 입력하세요: ").strip()
        
        if not file_path:
            print("❌ 파일 경로가 입력되지 않았습니다.")
            return
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        if file_path.suffix.lower() not in self.image_extensions:
            print(f"❌ 지원하지 않는 이미지 형식: {file_path.suffix}")
            return
        
        # 이미지 처리
        self._process_single_image(file_path)
    
    def _process_video_files(self):
        """동영상 파일 처리"""
        print("\n🎥 동영상 파일 처리")
        
        # 파일 경로 입력
        file_path = input("동영상 파일 경로를 입력하세요: ").strip()
        
        if not file_path:
            print("❌ 파일 경로가 입력되지 않았습니다.")
            return
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        if file_path.suffix.lower() not in self.video_extensions:
            print(f"❌ 지원하지 않는 동영상 형식: {file_path.suffix}")
            return
        
        # 동영상 처리
        self._process_single_video(file_path)
    
    def _process_upload_folder(self):
        """업로드 폴더 일괄 처리"""
        print(f"\n📂 업로드 폴더 일괄 처리: {self.upload_dir}")
        
        # 업로드 폴더 스캔
        all_files = []
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.image_extensions or ext in self.video_extensions:
                    all_files.append(file_path)
        
        if not all_files:
            print(f"❌ {self.upload_dir}에 처리할 파일이 없습니다.")
            return
        
        print(f"📋 발견된 파일: {len(all_files)}개")
        
        for i, file_path in enumerate(all_files, 1):
            print(f"\n🔄 [{i}/{len(all_files)}] 처리 중: {file_path.name}")
            
            try:
                if file_path.suffix.lower() in self.image_extensions:
                    self._process_single_image(file_path)
                else:
                    self._process_single_video(file_path)
            except Exception as e:
                logger.error(f"파일 처리 실패 {file_path}: {str(e)}")
                print(f"❌ 파일 처리 실패: {str(e)}")
    
    def _process_single_image(self, image_path: Path):
        """단일 이미지 처리"""
        print(f"\n🔍 이미지 분석 중: {image_path.name}")
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return
        
        # 얼굴 검출
        detection_result = self.detection_service.detect_faces(image)
        detected_faces = detection_result.faces
        
        if not detected_faces:
            print("❌ 검출된 얼굴이 없습니다.")
            return
        
        print(f"✅ {len(detected_faces)}개의 얼굴이 검출되었습니다.")
        
        # 사용자에게 얼굴 표시
        self._show_detected_faces(image, detected_faces, image_path.stem)
        
        # 사용자 선택 처리
        self._handle_face_selection(image, detected_faces, str(image_path))
    
    def _process_single_video(self, video_path: Path):
        """단일 동영상 처리"""
        print(f"\n🎥 동영상 분석 중: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 동영상 로드 실패: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 동영상 정보: {total_frames}프레임, {fps:.1f}FPS")
        
        # 프레임 샘플링 (매 30프레임마다)
        frame_skip = 30
        all_detected_faces = []
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if frame_number % frame_skip == 0:
                # 얼굴 검출
                detection_result = self.detection_service.detect_faces(frame)
                detected_faces = detection_result.faces
                
                if detected_faces:
                    for face in detected_faces:
                        face_info = {
                            'frame': frame.copy(),
                            'face': face,
                            'frame_number': frame_number,
                            'timestamp': frame_number / fps
                        }
                        all_detected_faces.append(face_info)
                
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    progress = frame_number / total_frames * 100
                    print(f"   진행률: {progress:.1f}% - 검출된 얼굴: {len(all_detected_faces)}개")
        
        cap.release()
        
        if not all_detected_faces:
            print("❌ 동영상에서 얼굴이 검출되지 않았습니다.")
            return
        
        print(f"✅ 총 {len(all_detected_faces)}개의 얼굴이 검출되었습니다.")
        
        # 품질 기준으로 정렬 및 샘플링
        self._process_video_faces(all_detected_faces, video_path.stem)
    
    def _show_detected_faces(self, image: np.ndarray, faces: List, source_name: str):
        """검출된 얼굴들을 시각적으로 표시"""
        display_image = image.copy()
        
        # 모든 얼굴에 바운딩 박스와 번호 표시
        for i, face in enumerate(faces):
            x, y, w, h = face.bbox.to_list()
            
            # 바운딩 박스
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 번호 표시
            cv2.putText(display_image, str(i + 1), (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # 신뢰도 표시
            cv2.putText(display_image, f"{face.confidence.value:.2f}", (x + 10, y + h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 윈도우 크기 조정
        height, width = display_image.shape[:2]
        if height > 800 or width > 1200:
            scale = min(800/height, 1200/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        cv2.imshow(f'검출된 얼굴 - {source_name}', display_image)
        print("🖼️ 얼굴 확인창이 열렸습니다. 아무 키나 눌러 계속...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _handle_face_selection(self, image: np.ndarray, faces: List, source_path: str):
        """얼굴 선택 및 이름 지정 처리"""
        print("\n👤 얼굴 선택 및 이름 지정")
        print("=" * 30)
        
        for i, face in enumerate(faces):
            print(f"{i + 1}. 얼굴 #{i + 1} (신뢰도: {face.confidence.value:.3f})")
        
        # 처리할 얼굴 선택
        while True:
            try:
                selections = input(f"\n처리할 얼굴 번호를 입력하세요 (1-{len(faces)}, 여러 개는 쉼표로 구분, 전체: all): ").strip()
                
                if selections.lower() == 'all':
                    selected_indices = list(range(len(faces)))
                    break
                elif selections:
                    selected_indices = [int(x.strip()) - 1 for x in selections.split(',')]
                    if all(0 <= idx < len(faces) for idx in selected_indices):
                        break
                    else:
                        print(f"❌ 잘못된 번호입니다. 1-{len(faces)} 범위의 번호를 입력하세요.")
                else:
                    print("❌ 선택을 입력해주세요.")
            except ValueError:
                print("❌ 올바른 번호를 입력하세요.")
        
        # 선택된 얼굴들 처리
        for idx in selected_indices:
            face = faces[idx]
            print(f"\n📷 얼굴 #{idx + 1} 처리 중...")
            
            # 인물 이름 입력
            person_name = input(f"이 얼굴의 인물 이름을 입력하세요: ").strip()
            
            if not person_name:
                print("❌ 이름이 입력되지 않았습니다. 건너뜁니다.")
                continue
            
            # 🎯 핵심: temp/face_staging로 이동 (기존 시스템과 동일)
            self._move_to_face_staging(image, face, person_name, source_path, idx)
    
    def _move_to_face_staging(self, image: np.ndarray, face, person_name: str, source_path: str, face_index: int):
        """🎯 공통 허브로 이동: temp/face_staging (기존 시스템과 동일)"""
        timestamp = int(time.time())
        
        # 전체 이미지 저장
        frame_filename = f"{person_name}_{timestamp}_frame_{face_index}.jpg"
        frame_path = self.face_staging_dir / frame_filename
        cv2.imwrite(str(frame_path), image)
        
        # 얼굴 크롭 저장
        x, y, w, h = face.bbox.to_list()
        face_crop = image[y:y+h, x:x+w]
        face_filename = f"{person_name}_{timestamp}_face_{face_index}.jpg"
        face_path = self.face_staging_dir / face_filename
        cv2.imwrite(str(face_path), face_crop)
        
        # 메타데이터 저장
        metadata = {
            'person_name': person_name,
            'timestamp': timestamp,
            'face_index': face_index,
            'bbox': face.bbox.to_list(),
            'confidence': face.confidence.value,
            'source_path': source_path,
            'source_type': 'batch_upload',
            'frame_path': str(frame_path),
            'face_path': str(face_path),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_filename = f"{person_name}_{timestamp}_meta_{face_index}.json"
        metadata_path = self.face_staging_dir / metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 공통 허브로 이동 완료: {face_filename}")
        print(f"   📂 저장 위치: {self.face_staging_dir}")
        print(f"   👤 인물 이름: {person_name}")
        print("   🎯 이제 기존 시스템과 동일한 분기 처리가 가능합니다!")
        
        return face_path
    
    def _process_video_faces(self, face_infos: List[Dict], video_name: str):
        """동영상에서 검출된 얼굴들 처리"""
        print(f"\n🎥 동영상 '{video_name}'에서 검출된 얼굴 처리")
        
        # 품질 기준으로 정렬
        face_infos.sort(key=lambda x: x['face'].confidence.value, reverse=True)
        
        # 상위 얼굴들만 표시 (최대 20개)
        display_count = min(20, len(face_infos))
        
        print(f"📋 상위 {display_count}개 얼굴 (품질 순):")
        for i in range(display_count):
            face_info = face_infos[i]
            print(f"  {i+1}. 프레임 {face_info['frame_number']} "
                  f"({face_info['timestamp']:.1f}초) - "
                  f"신뢰도: {face_info['face'].confidence.value:.3f}")
        
        # 대표 얼굴들 시각적 표시
        self._show_video_sample_faces(face_infos[:display_count], video_name)
        
        # 사용자 선택 처리
        while True:
            try:
                selections = input(f"\n처리할 얼굴 번호를 입력하세요 (1-{display_count}, 여러 개는 쉼표로 구분): ").strip()
                
                if not selections:
                    print("❌ 선택을 입력해주세요.")
                    continue
                
                selected_indices = [int(x.strip()) - 1 for x in selections.split(',')]
                if all(0 <= idx < display_count for idx in selected_indices):
                    break
                else:
                    print(f"❌ 잘못된 번호입니다. 1-{display_count} 범위의 번호를 입력하세요.")
            except ValueError:
                print("❌ 올바른 번호를 입력하세요.")
        
        # 인물 이름 입력
        person_name = input(f"선택된 얼굴들의 인물 이름을 입력하세요: ").strip()
        
        if not person_name:
            print("❌ 이름이 입력되지 않았습니다.")
            return
        
        # 선택된 얼굴들을 공통 허브로 이동
        for idx in selected_indices:
            face_info = face_infos[idx]
            self._move_to_face_staging(
                face_info['frame'], 
                face_info['face'], 
                person_name, 
                f"video_frame_{face_info['frame_number']}", 
                idx
            )
    
    def _show_video_sample_faces(self, face_infos: List[Dict], video_name: str):
        """동영상 샘플 얼굴들 표시"""
        if not face_infos:
            return
        
        # 그리드로 표시할 수 있는 만큼만
        display_count = min(12, len(face_infos))
        
        # 각 얼굴을 150x150으로 리사이즈
        face_images = []
        for i in range(display_count):
            face_info = face_infos[i]
            face = face_info['face']
            frame = face_info['frame']
            
            x, y, w, h = face.bbox.to_list()
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (150, 150))
                
                # 정보 오버레이
                cv2.putText(face_resized, f"#{i+1}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(face_resized, f"F{face_info['frame_number']}", (5, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(face_resized, f"{face.confidence.value:.2f}", (5, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                face_images.append(face_resized)
        
        if face_images:
            # 그리드로 배치 (4x3)
            cols = 4
            rows = (len(face_images) + cols - 1) // cols
            
            grid_width = cols * 150
            grid_height = rows * 150
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, face_img in enumerate(face_images):
                row = i // cols
                col = i % cols
                y_start = row * 150
                x_start = col * 150
                grid_image[y_start:y_start+150, x_start:x_start+150] = face_img
            
            cv2.imshow(f'동영상 얼굴 샘플 - {video_name}', grid_image)
            print("🖼️ 동영상 얼굴 샘플창이 열렸습니다. 아무 키나 눌러 계속...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    """메인 함수"""
    # 로깅 설정
    setup_logging()
    
    print("🎯 배치 얼굴 처리 시스템")
    print("사진/동영상 업로드 → 얼굴 검출 → 공통 처리 (temp/face_staging)")
    print("=" * 60)
    
    try:
        processor = BatchFaceProcessor()
        processor.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {str(e)}")
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 