#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 얼굴 캡처 시스템 v2

GUI 기반 얼굴 관리자가 포함된 고급 얼굴 캡처 시스템입니다.
- 카메라 화면이 멈추지 않는 삭제/편집 기능
- 미리보기 기능
- 이름 지정 및 그룹 지정 기능
- 개선된 얼굴 인식 성능
"""

import os
import sys
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from common.config import load_config

class FaceManagerGUI:
    """GUI 기반 얼굴 관리자"""
    
    def __init__(self, paths: Dict[str, Path]):
        self.paths = paths
        self.window = None
        self.selected_files = []
        
    def open_face_manager(self):
        """얼굴 관리 창 열기"""
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return
            
        self.window = tk.Toplevel()
        self.window.title("얼굴 관리자")
        self.window.geometry("1000x700")
        self.window.configure(bg='white')
        
        # 메인 프레임
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 좌측: 폴더 트리
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(left_frame, text="폴더 선택", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # 폴더 리스트박스
        self.folder_listbox = tk.Listbox(left_frame, width=25, height=15)
        self.folder_listbox.pack(fill=tk.Y)
        self.folder_listbox.bind('<<ListboxSelect>>', self.on_folder_select)
        
        # 폴더 목록 추가
        folders = {
            'detected_manual': '수동 캡처',
            'detected_auto': '자동 수집',
            'staging_named': '이름 지정됨'
        }
        
        for key, display_name in folders.items():
            file_count = len(list(self.paths[key].glob('*.jpg')))
            self.folder_listbox.insert(tk.END, f"{display_name} ({file_count}개)")
        
        # 우측: 이미지 그리드
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="얼굴 이미지", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # 스크롤 가능한 캔버스
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 하단: 버튼들
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="선택 삭제", command=self.delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="이름 지정", command=self.assign_name).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="그룹 지정", command=self.assign_group).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="새로고침", command=self.refresh_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="닫기", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
        
        self.image_widgets = []
        
    def on_folder_select(self, event):
        """폴더 선택 시 이미지 로드"""
        selection = self.folder_listbox.curselection()
        if not selection:
            return
            
        folder_index = selection[0]
        folder_keys = list(['detected_manual', 'detected_auto', 'staging_named'])
        
        if folder_index < len(folder_keys):
            self.current_folder = folder_keys[folder_index]
            self.load_images()
    
    def load_images(self):
        """현재 폴더의 이미지들 로드"""
        # 기존 위젯들 제거
        for widget in self.image_widgets:
            widget.destroy()
        self.image_widgets.clear()
        self.selected_files.clear()
        
        folder_path = self.paths[self.current_folder]
        image_files = list(folder_path.glob('*.jpg'))
        
        if not image_files:
            ttk.Label(self.scrollable_frame, text="이미지가 없습니다.").pack(pady=20)
            return
        
        # 그리드 레이아웃으로 이미지 표시
        cols = 5
        for i, img_file in enumerate(image_files):
            row = i // cols
            col = i % cols
            
            # 이미지 프레임
            img_frame = ttk.Frame(self.scrollable_frame)
            img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            try:
                # 이미지 로드 및 리사이즈
                pil_image = Image.open(img_file)
                pil_image.thumbnail((120, 120), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 체크박스
                var = tk.BooleanVar()
                checkbox = ttk.Checkbutton(img_frame, variable=var)
                checkbox.pack()
                
                # 이미지 라벨
                img_label = tk.Label(img_frame, image=photo, bg='white')
                img_label.image = photo  # 참조 유지
                img_label.pack()
                
                # 파일명 라벨
                filename_label = ttk.Label(img_frame, text=img_file.name[:20], font=('Arial', 8))
                filename_label.pack()
                
                self.image_widgets.extend([img_frame, checkbox, img_label, filename_label])
                self.selected_files.append((img_file, var))
                
            except Exception as e:
                print(f"이미지 로드 오류: {img_file.name} - {e}")
    
    def delete_selected(self):
        """선택된 이미지들 삭제"""
        selected = [f for f, var in self.selected_files if var.get()]
        
        if not selected:
            messagebox.showwarning("선택 없음", "삭제할 이미지를 선택하세요.")
            return
        
        if messagebox.askyesno("삭제 확인", f"{len(selected)}개 이미지를 삭제하시겠습니까?"):
            for img_file in selected:
                try:
                    img_file.unlink()
                    # 메타데이터도 삭제
                    metadata_file = img_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                except Exception as e:
                    print(f"삭제 오류: {img_file.name} - {e}")
            
            messagebox.showinfo("완료", f"{len(selected)}개 이미지가 삭제되었습니다.")
            self.load_images()
    
    def assign_name(self):
        """선택된 이미지들에 이름 지정"""
        selected = [f for f, var in self.selected_files if var.get()]
        
        if not selected:
            messagebox.showwarning("선택 없음", "이름을 지정할 이미지를 선택하세요.")
            return
        
        name = simpledialog.askstring("이름 입력", "인물 이름을 입력하세요:")
        if not name:
            return
        
        # staging/named/{name} 폴더에 이동
        target_dir = self.paths['staging_named'] / name
        target_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        for img_file in selected:
            try:
                new_path = target_dir / img_file.name
                img_file.rename(new_path)
                
                # 메타데이터도 이동
                metadata_file = img_file.with_suffix('.json')
                if metadata_file.exists():
                    new_metadata_path = target_dir / metadata_file.name
                    metadata_file.rename(new_metadata_path)
                
                moved_count += 1
            except Exception as e:
                print(f"이동 오류: {img_file.name} - {e}")
        
        messagebox.showinfo("완료", f"{moved_count}개 이미지가 '{name}' 폴더로 이동되었습니다.")
        self.load_images()
        self.refresh_folder_counts()
    
    def assign_group(self):
        """선택된 이미지들에 그룹 지정"""
        selected = [f for f, var in self.selected_files if var.get()]
        
        if not selected:
            messagebox.showwarning("선택 없음", "그룹을 지정할 이미지를 선택하세요.")
            return
        
        group_name = simpledialog.askstring("그룹 입력", "그룹 이름을 입력하세요:")
        if not group_name:
            return
        
        # staging/grouped/{group_name} 폴더에 이동
        target_dir = self.paths.get('staging_grouped')
        if not target_dir:
            # staging_grouped 경로 추가
            self.paths['staging_grouped'] = self.paths['staging_named'].parent / 'grouped'
            target_dir = self.paths['staging_grouped']
        
        target_dir = target_dir / group_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        for img_file in selected:
            try:
                new_path = target_dir / img_file.name
                img_file.rename(new_path)
                
                # 메타데이터도 이동
                metadata_file = img_file.with_suffix('.json')
                if metadata_file.exists():
                    new_metadata_path = target_dir / metadata_file.name
                    metadata_file.rename(new_metadata_path)
                
                moved_count += 1
            except Exception as e:
                print(f"이동 오류: {img_file.name} - {e}")
        
        messagebox.showinfo("완료", f"{moved_count}개 이미지가 '{group_name}' 그룹으로 이동되었습니다.")
        self.load_images()
    
    def refresh_view(self):
        """화면 새로고침"""
        self.refresh_folder_counts()
        if hasattr(self, 'current_folder'):
            self.load_images()
    
    def refresh_folder_counts(self):
        """폴더 개수 새로고침"""
        self.folder_listbox.delete(0, tk.END)
        folders = {
            'detected_manual': '수동 캡처',
            'detected_auto': '자동 수집',
            'staging_named': '이름 지정됨'
        }
        
        for key, display_name in folders.items():
            file_count = len(list(self.paths[key].glob('*.jpg')))
            self.folder_listbox.insert(tk.END, f"{display_name} ({file_count}개)")

class ImprovedFaceDetector:
    """개선된 얼굴 검출기"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.model_type = model_config.get('type', 'haar')
        self.confidence_threshold = 0.5
        
        # Haar Cascade 설정 개선
        self.scale_factor = 1.1  # 더 세밀한 검출
        self.min_neighbors = 3   # 더 민감한 검출
        self.min_size = (30, 30)  # 더 작은 얼굴도 검출
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        if self.model_type in ['haar', 'haar_default']:
            self.detector = cv2.CascadeClassifier(self.model_config['path'])
            if self.detector.empty():
                # 기본 Haar Cascade 사용
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 프로파일 얼굴 검출기도 추가
        self.profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """개선된 얼굴 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 히스토그램 평활화로 조명 개선
        gray = cv2.equalizeHist(gray)
        
        detections = []
        
        # 정면 얼굴 검출
        frontal_faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in frontal_faces:
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0,
                'type': 'frontal',
                'quality_score': self._calculate_quality_score(x, y, w, h, image.shape)
            })
        
        # 프로파일 얼굴 검출 (정면에서 검출되지 않은 경우)
        if len(detections) == 0:
            profile_faces = self.profile_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=self.min_size
            )
            
            for (x, y, w, h) in profile_faces:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # 프로파일은 신뢰도 낮게
                    'type': 'profile',
                    'quality_score': self._calculate_quality_score(x, y, w, h, image.shape)
                })
        
        return detections
    
    def _calculate_quality_score(self, x: int, y: int, w: int, h: int, image_shape: Tuple) -> float:
        """얼굴 품질 점수 계산"""
        # 크기 점수 (더 큰 얼굴이 높은 점수)
        face_area = w * h
        image_area = image_shape[0] * image_shape[1]
        size_ratio = face_area / image_area if image_area > 0 else 0
        size_score = min(size_ratio * 20, 1.0)  # 더 높은 가중치
        
        # 위치 점수 (중앙에 가까울수록 높은 점수)
        center_x, center_y = x + w//2, y + h//2
        img_center_x, img_center_y = image_shape[1]//2, image_shape[0]//2
        distance = ((center_x - img_center_x)**2 + (center_y - img_center_y)**2)**0.5
        max_distance = (img_center_x**2 + img_center_y**2)**0.5
        position_score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        # 종횡비 점수 (정사각형에 가까울수록 높은 점수)
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)
        
        return (size_score * 0.5 + position_score * 0.3 + aspect_score * 0.2)

class EnhancedFaceCaptureSystemV2:
    """개선된 얼굴 캡처 시스템 v2"""
    
    def __init__(self, camera_id: int = 0):
        # 로깅 설정
        setup_logging()
        self.logger = get_logger(__name__)
        
        # 카메라 설정
        self.camera_id = camera_id
        self.cap = None
        
        # 모델 설정 (개선된 Haar Cascade 사용)
        model_config = {
            'name': 'Improved Haar Cascade',
            'type': 'haar',
            'path': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        }
        
        # 개선된 얼굴 검출기
        self.face_detector = ImprovedFaceDetector(model_config)
        
        # GUI 관리자
        self.face_manager = None
        
        # 경로 설정
        self.domain_root = project_root / 'data' / 'domains' / 'face_recognition'
        self.paths = {
            'raw_captured': self.domain_root / 'raw_input' / 'captured',
            'detected_manual': self.domain_root / 'detected_faces' / 'from_manual',
            'detected_auto': self.domain_root / 'detected_faces' / 'auto_collected',
            'staging_named': self.domain_root / 'staging' / 'named',
            'staging_grouped': self.domain_root / 'staging' / 'grouped'
        }
        
        # 폴더 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # 상태 변수
        self.is_auto_mode = False
        self.show_info = True
        self.current_fps = 0
        self.fps_counter = 0
        self.fps_timer = time.time()
        
        # 세션 통계
        self.session_stats = {
            'auto_saved': 0,
            'manual_captured': 0,
            'named_saved': 0
        }
        
        self.logger.info("✅ 개선된 얼굴 캡처 시스템 v2 초기화 완료")
    
    def _adjust_confidence_threshold(self, delta: float):
        """신뢰도 임계값 조절"""
        self.face_detector.confidence_threshold = max(0.1, min(1.0, self.face_detector.confidence_threshold + delta))
        print(f"🎯 신뢰도 임계값: {self.face_detector.confidence_threshold:.2f}")
    
    def _adjust_scale_factor(self, delta: float):
        """스케일 팩터 조절"""
        self.face_detector.scale_factor = max(1.01, min(2.0, self.face_detector.scale_factor + delta))
        print(f"📏 스케일 팩터: {self.face_detector.scale_factor:.2f}")
    
    def _adjust_min_neighbors(self, delta: int):
        """최소 이웃 수 조절"""
        self.face_detector.min_neighbors = max(1, min(10, self.face_detector.min_neighbors + delta))
        print(f"👥 최소 이웃 수: {self.face_detector.min_neighbors}")
    
    def _switch_to_next_model(self):
        """다음 모델로 변경"""
        self._switch_model(1)
    
    def _switch_to_previous_model(self):
        """이전 모델로 변경"""
        self._switch_model(-1)
    
    def _switch_model(self, direction: int):
        """모델 전환"""
        # 현재는 Haar Cascade만 사용하므로 메시지만 출력
        print("🔄 현재는 개선된 Haar Cascade 모델만 사용 중입니다")
        print("   다른 모델을 사용하려면 모델 파일을 추가하세요")
    
    def start_capture(self):
        """캡처 시작"""
        try:
            if not self._initialize_camera():
                return
            
            self.logger.info("🚀 개선된 얼굴 캡처 시스템 v2 시작")
            self._print_help()
            self._capture_loop()
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        except Exception as e:
            self.logger.error(f"캡처 오류: {e}")
        finally:
            self._cleanup()
    
    def _initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"카메라 {self.camera_id} 연결 실패")
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ 카메라 {self.camera_id} 연결 성공 ({width}x{height})")
            return True
            
        except Exception as e:
            self.logger.error(f"카메라 초기화 오류: {e}")
            return False
    
    def _capture_loop(self):
        """메인 캡처 루프"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("프레임 읽기 실패")
                break
            
            # 얼굴 검출
            detections = self.face_detector.detect_faces(frame)
            
            # 자동 모드에서 얼굴 자동 저장
            if self.is_auto_mode and detections:
                self._handle_auto_save(frame, detections)
            
            # 시각화
            display_frame = self._visualize_detections(frame, detections)
            
            if self.show_info:
                display_frame = self._draw_info_overlay(display_frame, detections)
            
            cv2.imshow('Enhanced Face Capture v2', display_frame)
            
            # FPS 업데이트
            self._update_fps()
            
            # 키보드 입력 처리
            action = self._handle_keyboard_input(cv2.waitKey(1) & 0xFF, frame, detections)
            if action == 'quit':
                break
    
    def _handle_keyboard_input(self, key: int, frame: np.ndarray, detections: List[Dict]) -> str:
        """키보드 입력 처리"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('i'):
            self.show_info = not self.show_info
            status = "ON" if self.show_info else "OFF"
            print(f"ℹ️  정보 표시: {status}")
        elif key == ord('a'):
            self.is_auto_mode = not self.is_auto_mode
            mode = "🤖 자동" if self.is_auto_mode else "👤 수동"
            print(f"🔄 모드 변경: {mode}")
        elif key == ord('s'):
            self._save_full_frame(frame)
        elif key == ord('c'):
            self._handle_manual_face_capture(frame, detections)
        elif key == ord('e'):  # 편집 모드 (새로운 기능)
            self._open_face_manager()
        elif key == ord('h'):
            self._print_help()
        # 모델 설정 조절 키 추가
        elif key == ord('+') or key == ord('='):
            self._adjust_confidence_threshold(0.05)
        elif key == ord('-') or key == ord('_'):
            self._adjust_confidence_threshold(-0.05)
        elif key == ord('['):
            self._adjust_scale_factor(-0.05)
        elif key == ord(']'):
            self._adjust_scale_factor(0.05)
        elif key == ord(','):
            self._adjust_min_neighbors(-1)
        elif key == ord('.'):
            self._adjust_min_neighbors(1)
        # 모델 변경 키 추가
        elif key == ord('n'):
            self._switch_to_next_model()
        elif key == ord('m'):
            self._switch_to_previous_model()
        
        return 'continue'
    
    def _open_face_manager(self):
        """얼굴 관리자 열기"""
        def open_manager():
            try:
                face_manager = FaceManagerGUI(self.paths)
                face_manager.open_face_manager()
                print("\n" + "="*60)
                print("🎯 얼굴 관리자 사용법")
                print("="*60)
                print("📁 폴더 선택:")
                print("   - from_manual: 수동으로 캡처한 얼굴들")
                print("   - from_captured: 전체 프레임에서 검출된 얼굴들")
                print("   - from_uploads: 업로드된 이미지에서 검출된 얼굴들")
                print("   - auto_collected: 자동 모드에서 저장된 얼굴들")
                print()
                print("🖱️  마우스 조작:")
                print("   - 클릭: 이미지 선택/해제")
                print("   - Ctrl+클릭: 여러 이미지 선택")
                print("   - Shift+클릭: 범위 선택")
                print()
                print("🔧 기능 버튼:")
                print("   - [Delete Selected]: 선택된 이미지들 삭제")
                print("   - [Assign Name]: 이름 지정 (staging/named/ 폴더로 이동)")
                print("   - [Assign Group]: 그룹 지정 (staging/grouped/ 폴더로 이동)")
                print("   - [Refresh]: 목록 새로고침")
                print()
                print("💡 팁:")
                print("   - 이름 지정 시: staging/named/{이름}/ 폴더에 저장")
                print("   - 그룹 지정 시: staging/grouped/{그룹}/ 폴더에 저장")
                print("   - 삭제 시: 영구적으로 삭제되므로 주의")
                print("="*60)
            except Exception as e:
                print(f"얼굴 관리자 오류: {e}")
        
        # 별도 스레드에서 실행
        manager_thread = threading.Thread(target=open_manager, daemon=True)
        manager_thread.start()
    
    def _save_full_frame(self, frame: np.ndarray):
        """전체 프레임 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        frame_filename = f"captured_frame_{timestamp}.jpg"
        frame_path = self.paths['raw_captured'] / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        metadata = {
            'timestamp': timestamp,
            'capture_type': 'manual_frame',
            'frame_path': str(frame_path),
            'frame_size': list(frame.shape)
        }
        
        metadata_path = frame_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 전체 프레임 저장: {frame_filename}")
        self.session_stats['manual_captured'] += 1
    
    def _handle_manual_face_capture(self, frame: np.ndarray, detections: List[Dict]):
        """수동 얼굴 캡처"""
        if not detections:
            print("❌ 검출된 얼굴이 없습니다")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            face_crop = frame[y:y+h, x:x+w]
            
            face_filename = f"manual_face_{timestamp}_{i:02d}_conf{detection['confidence']:.2f}.jpg"
            face_path = self.paths['detected_manual'] / face_filename
            cv2.imwrite(str(face_path), face_crop)
            
            metadata = {
                'timestamp': timestamp,
                'capture_type': 'manual',
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(detection['confidence']),
                'quality_score': float(detection['quality_score']),
                'face_type': detection.get('type', 'frontal')
            }
            
            metadata_path = face_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {len(detections)}개 얼굴을 detected_faces/from_manual에 저장")
        self.session_stats['manual_captured'] += len(detections)
    
    def _handle_auto_save(self, frame: np.ndarray, detections: List[Dict]):
        """자동 모드에서 얼굴 저장"""
        for detection in detections:
            if detection['quality_score'] > 0.3:  # 품질 임계값
                self._auto_save_detected_face(frame, detection)
    
    def _auto_save_detected_face(self, frame: np.ndarray, detection: Dict):
        """자동 얼굴 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        bbox = detection['bbox']
        x, y, w, h = bbox
        face_crop = frame[y:y+h, x:x+w]
        
        face_filename = f"auto_face_{timestamp}_conf{detection['confidence']:.2f}_qual{detection['quality_score']:.2f}.jpg"
        face_path = self.paths['detected_auto'] / face_filename
        cv2.imwrite(str(face_path), face_crop)
        
        metadata = {
            'timestamp': timestamp,
            'capture_type': 'auto',
            'bbox': [int(x), int(y), int(w), int(h)],
            'confidence': float(detection['confidence']),
            'quality_score': float(detection['quality_score']),
            'face_type': detection.get('type', 'frontal')
        }
        
        metadata_path = face_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.session_stats['auto_saved'] += 1
    
    def _visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """검출 결과 시각화"""
        display_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            confidence = detection['confidence']
            quality = detection['quality_score']
            face_type = detection.get('type', 'frontal')
            
            # 얼굴 타입에 따른 색상
            color = (0, 255, 0) if face_type == 'frontal' else (0, 255, 255)
            
            # 바운딩 박스
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # 정보 텍스트
            info_text = f"{face_type} {confidence:.2f} Q:{quality:.2f}"
            cv2.putText(display_frame, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_frame
    
    def _draw_info_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """정보 오버레이"""
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # 상단 상태 정보 (영문으로 변경)
        mode_text = "AUTO" if self.is_auto_mode else "MANUAL"
        cv2.putText(overlay_frame, f"Mode: {mode_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(overlay_frame, f"FPS: {self.current_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(overlay_frame, f"Faces: {len(detections)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(overlay_frame, f"Auto: {self.session_stats['auto_saved']}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 모델 정보 (영문으로 변경)
        model_name = self.face_detector.model_config.get('name', 'Unknown')
        # 모델 이름이 길면 축약
        if len(model_name) > 30:
            model_name = model_name[:27] + "..."
        cv2.putText(overlay_frame, f"Model: {model_name}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 하단 토글키 정보
        self._draw_toggle_keys(overlay_frame, width, height)
        
        return overlay_frame
    
    def _draw_toggle_keys(self, frame: np.ndarray, width: int, height: int):
        """토글키 정보를 화면 하단에 표시"""
        # 배경 박스 (반투명)
        box_height = 140
        box_y = height - box_height - 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, box_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 토글키 정보 (영문으로 변경)
        y_start = box_y + 20
        line_height = 25
        
        # 첫 번째 줄: 기본 명령어
        cv2.putText(frame, "Toggle Keys:", (10, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 두 번째 줄: 모드 및 정보
        cv2.putText(frame, "A: Auto/Manual  I: Info  H: Help  Q: Quit", (10, y_start + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 세 번째 줄: 캡처 명령어
        cv2.putText(frame, "S: Save Frame  C: Capture Face  E: Face Manager", (10, y_start + line_height * 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 네 번째 줄: 모델 설정
        cv2.putText(frame, "Model Settings: +/- Confidence  [/] Scale  ,/. Neighbors  N/M: Model", (10, y_start + line_height * 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 다섯 번째 줄: 현재 설정값
        current_settings = f"Current: Conf={self.face_detector.confidence_threshold:.2f} Scale={self.face_detector.scale_factor:.2f} Neighbors={self.face_detector.min_neighbors}"
        cv2.putText(frame, current_settings, (10, y_start + line_height * 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _put_korean_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], font_size: int = 20, color: Tuple[int, int, int] = (255, 255, 255)):
        """한글 텍스트를 이미지에 추가 (PIL 사용)"""
        try:
            # OpenCV 이미지를 PIL 이미지로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 폰트 설정 (기본 폰트 사용)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # 텍스트 그리기
            draw = ImageDraw.Draw(pil_image)
            draw.text(position, text, font=font, fill=color)
            
            # PIL 이미지를 OpenCV 이미지로 변환
            frame_rgb = np.array(pil_image)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            return frame_bgr
        except Exception as e:
            # 오류 시 영문으로 대체
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return frame
    
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
        print("\n" + "="*60)
        print("📋 개선된 얼굴 캡처 시스템 v2 - 키보드 명령어")
        print("="*60)
        print("🔧 공통 명령어:")
        print("   i  - 정보 표시 토글")
        print("   a  - 자동/수동 모드 전환")
        print("   h  - 도움말 표시")
        print("   q  - 종료")
        print()
        print("👤 캡처 명령어:")
        print("   s  - 전체 프레임 저장 (raw_input/captured/)")
        print("   c  - 얼굴 캡처 (detected_faces/from_manual/)")
        print("   e  - 얼굴 관리자 열기 (미리보기, 편집, 삭제)")
        print()
        print("🎛️  모델 설정 조절:")
        print("   +/- - 신뢰도 임계값 조절")
        print("   [/] - 스케일 팩터 조절")
        print("   ,/. - 최소 이웃 수 조절")
        print("   n/m - 모델 변경 (사용 가능한 모델이 있을 때)")
        print()
        print("🎨 얼굴 관리자 기능:")
        print("   - 미리보기: 저장된 얼굴 이미지 확인")
        print("   - 이름 지정: staging/named/{이름}/ 폴더로 이동")
        print("   - 그룹 지정: staging/grouped/{그룹}/ 폴더로 이동")
        print("   - 삭제: 선택한 이미지들 삭제")
        print()
        print("📁 데이터 플로우:")
        print("   1. s키: raw_input/captured/ (전체 프레임)")
        print("   2. c키: detected_faces/from_manual/ (얼굴만)")
        print("   3. 자동모드: detected_faces/auto_collected/ (얼굴만)")
        print("   4. e키: 얼굴 관리자로 편집/이동/삭제")
        print("="*60)
    
    def _cleanup(self):
        """정리 작업"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n📊 세션 통계:")
        print(f"   자동 저장: {self.session_stats['auto_saved']}개")
        print(f"   수동 캡처: {self.session_stats['manual_captured']}개")
        
        self.logger.info("개선된 얼굴 캡처 시스템 v2 종료")

def main():
    """메인 함수"""
    try:
        system = EnhancedFaceCaptureSystemV2()
        system.start_capture()
    except Exception as e:
        print(f"시스템 오류: {e}")

if __name__ == "__main__":
    main() 