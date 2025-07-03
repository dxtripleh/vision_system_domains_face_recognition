#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 얼굴 캡처 시스템 테스트

요청사항 구현:
1. i키로 정보 토글 (카메라, 모델 정보 표시)
2. s키로 captured_frames 폴더에 저장 (자동 얼굴 검출 및 그룹핑)
3. a키로 자동/수동 모드 전환
"""

import os
import sys
import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

class SimpleFaceCaptureSystem:
    """간단한 얼굴 캡처 시스템"""
    
    def __init__(self, camera_id: int = 0):
        # 카메라 설정
        self.camera_id = camera_id
        self.cap = None
        
        # 경로 설정
        self.captured_frames_dir = project_root / 'data' / 'temp' / 'captured_frames'
        self.face_staging_dir = project_root / 'data' / 'temp' / 'face_staging'
        
        # 폴더 생성
        for directory in [self.captured_frames_dir, self.face_staging_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 모드 설정
        self.is_auto_mode = False  # 기본값: 수동 모드
        self.show_info = True
        
        # 얼굴 검출기 (OpenCV Haar Cascade)
        cascade_path = project_root / 'models' / 'weights' / 'face_detection_opencv_haarcascade_20250628.xml'
        if cascade_path.exists():
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        else:
            # 기본 OpenCV Haar Cascade 사용
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 성능 모니터링
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # 캡처 통계
        self.captured_faces = []
        
        print("✅ 간단한 얼굴 캡처 시스템 초기화 완료")
    
    def start_capture(self):
        """캡처 시작"""
        print("🚀 간단한 얼굴 캡처 시스템 시작")
        print("="*60)
        
        # 환경 정보 출력
        self._print_environment_info()
        
        # 카메라 초기화
        if not self._initialize_camera():
            return False
        
        print("\n📹 카메라 시작됨. 키보드 명령어:")
        self._print_help()
        
        try:
            # 메인 캡처 루프
            self._capture_loop()
            
        except KeyboardInterrupt:
            print("\n⚠️  사용자가 중단했습니다.")
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
        finally:
            self._cleanup()
        
        return True
    
    def _print_environment_info(self):
        """환경 정보 출력"""
        print(f"🔍 환경 정보:")
        print(f"   카메라 ID: {self.camera_id}")
        print(f"   검출 모델: OpenCV Haar Cascade")
        print(f"   모드: {'🤖 자동' if self.is_auto_mode else '👤 수동'}")
        print(f"   정보 표시: {'ON' if self.show_info else 'OFF'}")
    
    def _initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"❌ 카메라 {self.camera_id} 연결 실패")
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 테스트 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 카메라에서 프레임을 읽을 수 없습니다")
                return False
            
            print(f"✅ 카메라 {self.camera_id} 연결 성공 ({frame.shape[1]}x{frame.shape[0]})")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 초기화 실패: {str(e)}")
            return False
    
    def _capture_loop(self):
        """메인 캡처 루프"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패")
                break
            
            # 얼굴 검출
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # 검출 결과를 딕셔너리 형태로 변환
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8  # Haar Cascade는 신뢰도를 제공하지 않으므로 고정값
                })
            
            # 검출 결과 시각화
            display_frame = self._visualize_detections(frame, detections)
            
            # 정보 오버레이
            if self.show_info:
                display_frame = self._draw_info_overlay(display_frame, detections)
            
            # 간단한 키 정보 표시 (하단)
            display_frame = self._draw_key_info(display_frame)
            
            # 화면 표시 (고정된 창 이름 사용)
            window_title = "Face Capture System"
            cv2.imshow(window_title, display_frame)
            
            # 키보드 입력 처리
            action = self._handle_keyboard_input()
            if action == 'quit':
                break
            elif action == 'save_frame':
                self._save_frame_for_processing(frame, detections)
            elif action == 'capture_face':
                self._capture_face_with_name(frame, detections)
            elif action == 'toggle_info':
                self.show_info = not self.show_info
                print(f"💡 정보 표시: {'ON' if self.show_info else 'OFF'}")
            elif action == 'toggle_mode':
                self._toggle_auto_manual_mode()
            elif action == 'show_help':
                self._print_help()
            
            # FPS 계산
            self._update_fps()
    
    def _visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """검출 결과 시각화"""
        display_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # 바운딩 박스 그리기 (모드에 따라 색상 변경)
            if self.is_auto_mode:
                color = (255, 165, 0)  # 오렌지 (자동)
            else:
                color = (0, 255, 0)    # 초록 (수동)
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # 신뢰도 텍스트
            text = f"{confidence:.2f}"
            cv2.putText(display_frame, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_frame
    
    def _draw_info_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """정보 오버레이 그리기"""
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 정보 텍스트
        y_offset = 30
        line_height = 18
        
        texts = [
            f"Camera: {self.camera_id} | Model: OpenCV Haar Cascade",
            f"Mode: {'AUTO' if self.is_auto_mode else 'MANUAL'} | FPS: {self.current_fps:.1f}",
            f"Faces: {len(detections)} | Captured: {len(self.captured_faces)}",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def _draw_key_info(self, frame: np.ndarray) -> np.ndarray:
        """간단한 키 정보 표시 (하단)"""
        if not self.show_info:
            return frame
        
        # 하단에 간단한 키 정보 표시
        height, width = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 50), (width - 10, height - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # 키 정보 텍스트
        key_info = "i:Info | a:Auto/Manual | s:Save Frame | c:Capture | h:Help | q:Quit"
        
        cv2.putText(frame, key_info, (20, height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _handle_keyboard_input(self) -> Optional[str]:
        """키보드 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('i'):
            return 'toggle_info'
        elif key == ord('s'):
            return 'save_frame'
        elif key == ord('c'):
            return 'capture_face'
        elif key == ord('a'):
            return 'toggle_mode'
        elif key == ord('h'):
            return 'show_help'
        
        return None
    
    def _toggle_auto_manual_mode(self):
        """자동/수동 모드 전환"""
        self.is_auto_mode = not self.is_auto_mode
        mode_name = "자동" if self.is_auto_mode else "수동"
        icon = "🤖" if self.is_auto_mode else "👤"
        
        print(f"{icon} {mode_name} 모드로 전환")
        
        if self.is_auto_mode:
            print("   💡 자동 모드: s키로 프레임을 저장하면 자동 처리됩니다")
        else:
            print("   💡 수동 모드: c키로 얼굴을 수동 캡처합니다")
    
    def _save_frame_for_processing(self, frame: np.ndarray, detections: List[Dict]):
        """프레임을 captured_frames 폴더에 저장하고 자동 처리"""
        if not detections:
            print("⚠️  저장할 얼굴이 없습니다.")
            return
        
        # captured_frames 폴더에 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"captured_frame_{timestamp}.jpg"
        dest_path = self.captured_frames_dir / filename
        
        success = cv2.imwrite(str(dest_path), frame)
        if not success:
            print("❌ 프레임 저장 실패")
            return
        
        print(f"📸 프레임 저장: {dest_path}")
        print(f"🔍 {len(detections)}개 얼굴 발견, 자동 처리 시작...")
        
        # 간단한 자동 그룹핑 및 이름 지정
        self._process_saved_frame(frame, detections, timestamp)
    
    def _process_saved_frame(self, frame: np.ndarray, detections: List[Dict], timestamp: str):
        """저장된 프레임의 얼굴 자동 처리"""
        print(f"\n📊 {len(detections)}개 얼굴 처리 중...")
        
        # 얼굴 격자 생성 및 표시
        face_crops = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # 여유를 두고 얼굴 영역 추출
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crops.append(face_crop)
        
        if not face_crops:
            print("❌ 얼굴 추출 실패")
            return
        
        # 얼굴 격자 표시
        combined_image = self._create_face_grid(face_crops)
        cv2.imshow(f'Found {len(face_crops)} faces - Enter name in console', combined_image)
        cv2.waitKey(1)
        
        # 이름 입력 받기
        try:
            person_name = input(f"👤 {len(face_crops)}개 얼굴의 인물 이름을 입력하세요 (Enter=건너뛰기): ").strip()
            
            cv2.destroyWindow(f'Found {len(face_crops)} faces - Enter name in console')
            
            if person_name:
                self._save_faces_to_staging(person_name, face_crops, timestamp)
            else:
                print("⏭️  건너뛰기")
                
        except (KeyboardInterrupt, EOFError):
            cv2.destroyWindow(f'Found {len(face_crops)} faces - Enter name in console')
            print("⏭️  입력 취소")
    
    def _create_face_grid(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """얼굴들을 격자로 배열"""
        if not face_crops:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 격자 크기 계산
        num_faces = len(face_crops)
        cols = min(3, num_faces)
        rows = (num_faces + cols - 1) // cols
        
        # 각 얼굴을 동일한 크기로 리사이즈
        face_size = 150
        resized_faces = []
        
        for i, face_crop in enumerate(face_crops):
            resized = cv2.resize(face_crop, (face_size, face_size))
            
            # 번호 텍스트 추가
            cv2.putText(resized, f"#{i+1}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            resized_faces.append(resized)
        
        # 격자 생성
        grid_height = rows * face_size
        grid_width = cols * face_size
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, face in enumerate(resized_faces):
            row = i // cols
            col = i % cols
            
            y1 = row * face_size
            y2 = y1 + face_size
            x1 = col * face_size
            x2 = x1 + face_size
            
            grid[y1:y2, x1:x2] = face
        
        return grid
    
    def _save_faces_to_staging(self, person_name: str, face_crops: List[np.ndarray], timestamp: str):
        """얼굴들을 face_staging으로 저장"""
        # 안전한 파일명 생성
        safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        source = "auto_saved" if self.is_auto_mode else "manual_saved"
        folder_name = f"{safe_name}_{timestamp}_{source}"
        staging_dir = self.face_staging_dir / folder_name
        staging_dir.mkdir(exist_ok=True)
        
        # 메타데이터 생성
        metadata = {
            'person_name': person_name,
            'safe_name': safe_name,
            'created_at': timestamp,
            'source': source,
            'face_count': len(face_crops),
            'capture_mode': 'auto' if self.is_auto_mode else 'manual',
            'camera_id': self.camera_id
        }
        
        metadata_file = staging_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 얼굴들 저장
        saved_count = 0
        for i, face_crop in enumerate(face_crops):
            filename = f"face_{safe_name}_{timestamp}_{i:02d}.jpg"
            dest_path = staging_dir / filename
            
            success = cv2.imwrite(str(dest_path), face_crop)
            if success:
                saved_count += 1
        
        print(f"✅ {person_name}: {saved_count}개 얼굴을 face_staging으로 저장")
        print(f"📁 저장 위치: {staging_dir}")
        
        # 캡처 통계 업데이트
        self.captured_faces.extend([{'name': person_name, 'count': saved_count}])
    
    def _capture_face_with_name(self, frame: np.ndarray, detections: List[Dict]):
        """얼굴 캡처 및 이름 지정 (수동 모드 전용)"""
        if self.is_auto_mode:
            print("⚠️  자동 모드에서는 c키를 사용할 수 없습니다. a키로 수동 모드로 전환하세요.")
            return
        
        if not detections:
            print("⚠️  캡처할 얼굴이 없습니다.")
            return
        
        print(f"\n👤 {len(detections)}개 얼굴 발견됨")
        
        # 얼굴 영역들을 하이라이트한 프레임 생성
        highlight_frame = frame.copy()
        for detection in detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(highlight_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        cv2.putText(highlight_frame, "Enter person name in console", 
                   (20, highlight_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Face Capture - Enter name in console', highlight_frame)
        cv2.waitKey(1)
        
        # 이름 입력 받기
        try:
            person_name = input(f"👤 인물 이름을 입력하세요 (Enter=취소): ").strip()
            
            cv2.destroyWindow('Face Capture - Enter name in console')
            
            if person_name:
                # 얼굴 추출 및 저장
                face_crops = []
                for detection in detections:
                    x, y, w, h = detection['bbox']
                    margin = 20
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_crops.append(face_crop)
                
                if face_crops:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._save_faces_to_staging(person_name, face_crops, timestamp)
                    print(f"✅ {person_name}: {len(face_crops)}개 얼굴 캡처 완료")
                else:
                    print("❌ 얼굴 추출 실패")
            else:
                print("⏭️  캡처 취소됨")
                
        except (KeyboardInterrupt, EOFError):
            cv2.destroyWindow('Face Capture - Enter name in console')
            print("⏭️  입력 취소")
    
    def _update_fps(self):
        """FPS 업데이트"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def _print_help(self):
        """도움말 출력"""
        help_text = f"""
📋 간단한 얼굴 캡처 시스템 - 키보드 명령어:
   i  - 정보 표시 토글 (카메라, 모델 정보)
   a  - 자동/수동 모드 전환 (현재: {'자동' if self.is_auto_mode else '수동'})
   s  - 현재 프레임을 captured_frames에 저장 (자동 얼굴 처리)
   c  - 얼굴 캡처 + 이름 지정 (수동 모드만)
   h  - 도움말 표시
   q  - 종료

💡 사용법:
   🤖 자동 모드: s키로 프레임 저장 → 자동 얼굴 검출 및 그룹핑
   👤 수동 모드: c키로 수동 캡처 또는 s키로 프레임 저장 후 처리
        """
        print(help_text)
    
    def _cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print(f"\n📊 세션 요약:")
        print(f"   모드: {'자동' if self.is_auto_mode else '수동'}")
        print(f"   캡처된 얼굴: {len(self.captured_faces)}개")
        print("✅ 시스템 종료")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="간단한 얼굴 캡처 시스템")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID (기본값: 0)")
    parser.add_argument("--auto", action="store_true", help="자동 모드로 시작")
    args = parser.parse_args()
    
    # 하드웨어 연결 검증
    print("🔍 하드웨어 연결 상태 확인 중...")
    
    # 카메라 연결 테스트
    test_cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not test_cap.isOpened():
        print(f"❌ 카메라 {args.camera}에 연결할 수 없습니다.")
        print("💡 다른 카메라 ID를 시도해보세요: --camera 1")
        return
    
    ret, frame = test_cap.read()
    test_cap.release()
    
    if not ret:
        print(f"❌ 카메라 {args.camera}에서 영상을 읽을 수 없습니다.")
        return
    
    print(f"✅ 카메라 {args.camera} 연결 확인됨")
    
    # 시스템 시작
    capture_system = SimpleFaceCaptureSystem(camera_id=args.camera)
    
    # 자동 모드 설정
    if args.auto:
        capture_system.is_auto_mode = True
    
    capture_system.start_capture()

if __name__ == "__main__":
    main() 