#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 그룹 관리 도구.

자동으로 그룹핑된 얼굴들을 확인하고 이름을 지정하는 도구입니다.
"""

import os
import sys
import cv2
import json
import logging
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.core.services.face_grouping_service import FaceGroupingService
from domains.face_recognition.infrastructure.storage.file_storage import FileStorage

logger = logging.getLogger(__name__)


class FaceGroupManager:
    """얼굴 그룹 관리자"""
    
    def __init__(self):
        self.grouping_service = FaceGroupingService()
        self.file_storage = FileStorage()
        
    def show_group_statistics(self):
        """그룹 통계 표시"""
        stats = self.grouping_service.get_statistics()
        
        print("\n" + "="*50)
        print("📊 얼굴 그룹 통계")
        print("="*50)
        print(f"총 그룹 수: {stats['total_groups']}")
        print(f"이름 있는 그룹: {stats['named_groups']}")
        print(f"이름 없는 그룹: {stats['unnamed_groups']}")
        print(f"총 얼굴 수: {stats['total_faces']}")
        print(f"그룹당 평균 얼굴 수: {stats['avg_faces_per_group']:.1f}")
        print("="*50)
    
    def show_unnamed_groups(self):
        """이름이 없는 그룹들 표시"""
        unnamed_groups = self.grouping_service.get_ungrouped_faces()
        
        if not unnamed_groups:
            print("\n✅ 모든 그룹에 이름이 지정되어 있습니다!")
            return []
        
        print(f"\n📋 이름이 없는 그룹 ({len(unnamed_groups)}개)")
        print("-" * 50)
        
        for i, group in enumerate(unnamed_groups, 1):
            print(f"\n[{i}] 그룹 ID: {group.group_id}")
            print(f"    얼굴 수: {len(group.faces)}")
            print(f"    생성일: {self._format_timestamp(group.created_at)}")
            
            # 대표 얼굴 정보
            if group.representative_face:
                rep_face = group.representative_face
                print(f"    대표 얼굴: {rep_face.face_id}")
                print(f"    품질 점수: {rep_face.quality_score:.3f}")
        
        return unnamed_groups
    
    def _format_timestamp(self, timestamp: float) -> str:
        """타임스탬프 포맷팅"""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def show_group_faces(self, group):
        """그룹의 얼굴들을 시각적으로 표시"""
        print(f"\n👥 그룹 {group.group_id}의 얼굴들 ({len(group.faces)}개)")
        
        # 그리드로 얼굴들을 표시
        faces_per_row = 4
        rows = (len(group.faces) + faces_per_row - 1) // faces_per_row
        
        # 각 얼굴 이미지 로드 및 리사이즈
        face_images = []
        for face in group.faces:
            try:
                # 실제 얼굴 이미지 파일이 있다면 로드
                temp_dir = Path("data/temp/face_staging")
                face_files = list(temp_dir.glob(f"*{face.face_id}*"))
                
                if face_files:
                    img = cv2.imread(str(face_files[0]))
                    if img is not None:
                        img = cv2.resize(img, (150, 150))
                        # 얼굴 ID와 품질 점수 표시
                        cv2.putText(img, f"ID: {face.face_id[:8]}", (5, 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(img, f"Q: {face.quality_score:.2f}", (5, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        face_images.append(img)
                    else:
                        # 이미지 로드 실패시 검은 이미지
                        face_images.append(np.zeros((150, 150, 3), dtype=np.uint8))
                else:
                    # 파일이 없으면 검은 이미지
                    face_images.append(np.zeros((150, 150, 3), dtype=np.uint8))
                    
            except Exception as e:
                logger.warning(f"Error loading face image for {face.face_id}: {str(e)}")
                face_images.append(np.zeros((150, 150, 3), dtype=np.uint8))
        
        # 그리드 이미지 생성
        if face_images:
            grid_rows = []
            for row in range(rows):
                start_idx = row * faces_per_row
                end_idx = min(start_idx + faces_per_row, len(face_images))
                row_images = face_images[start_idx:end_idx]
                
                # 행이 부족하면 빈 이미지로 채움
                while len(row_images) < faces_per_row:
                    row_images.append(np.zeros((150, 150, 3), dtype=np.uint8))
                
                row_img = np.hstack(row_images)
                grid_rows.append(row_img)
            
            grid_img = np.vstack(grid_rows)
            
            # 창 이름에 그룹 정보 포함
            window_name = f"Group {group.group_id[:8]} - {len(group.faces)} faces"
            cv2.imshow(window_name, grid_img)
            
            print(f"📷 그룹 얼굴들이 '{window_name}' 창에 표시됩니다.")
            print("   아무 키나 누르면 창이 닫힙니다.")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
    
    def name_groups_interactively(self):
        """대화형으로 그룹들에 이름 지정"""
        unnamed_groups = self.show_unnamed_groups()
        
        if not unnamed_groups:
            return
        
        print(f"\n🏷️  그룹에 이름을 지정해보세요!")
        print("   'skip' 입력시 건너뛰기, 'quit' 입력시 종료")
        
        for i, group in enumerate(unnamed_groups, 1):
            print(f"\n--- 그룹 {i}/{len(unnamed_groups)} ---")
            
            # 그룹 얼굴들 표시
            show_faces = input(f"그룹의 얼굴들을 보시겠습니까? (y/n, 기본값: y): ").strip().lower()
            if show_faces in ['', 'y', 'yes']:
                self.show_group_faces(group)
            
            # 이름 입력 받기
            while True:
                group_name = input(f"그룹 이름을 입력하세요 (얼굴 수: {len(group.faces)}개): ").strip()
                
                if group_name.lower() == 'quit':
                    print("종료합니다.")
                    return
                elif group_name.lower() == 'skip':
                    print("이 그룹을 건너뜁니다.")
                    break
                elif group_name:
                    # 이름 설정
                    success = self.grouping_service.set_group_name(group.group_id, group_name)
                    if success:
                        print(f"✅ '{group_name}' 이름이 설정되었습니다!")
                        break
                    else:
                        print("❌ 이름 설정에 실패했습니다. 다시 시도해주세요.")
                else:
                    print("이름을 입력해주세요. (skip으로 건너뛰기 가능)")
    
    def show_all_groups(self):
        """모든 그룹 표시"""
        all_groups = self.grouping_service.get_all_groups()
        
        if not all_groups:
            print("\n📭 등록된 그룹이 없습니다.")
            return
        
        print(f"\n📂 전체 그룹 목록 ({len(all_groups)}개)")
        print("-" * 70)
        
        for i, group in enumerate(all_groups, 1):
            name = group.group_name or "이름 없음"
            print(f"[{i:2d}] {name:<20} | 얼굴 수: {len(group.faces):2d}개 | "
                  f"생성일: {self._format_timestamp(group.created_at)}")
    
    def manage_groups_menu(self):
        """그룹 관리 메뉴"""
        while True:
            print("\n" + "="*50)
            print("🗂️  얼굴 그룹 관리 도구")
            print("="*50)
            print("1. 그룹 통계 보기")
            print("2. 이름 없는 그룹들 보기")
            print("3. 그룹에 이름 지정하기")
            print("4. 모든 그룹 보기")
            print("5. 종료")
            print("-" * 50)
            
            choice = input("선택하세요 (1-5): ").strip()
            
            try:
                if choice == '1':
                    self.show_group_statistics()
                elif choice == '2':
                    self.show_unnamed_groups()
                elif choice == '3':
                    self.name_groups_interactively()
                elif choice == '4':
                    self.show_all_groups()
                elif choice == '5':
                    print("그룹 관리 도구를 종료합니다.")
                    break
                else:
                    print("올바른 번호를 입력해주세요.")
                    
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                logger.error(f"Error in group management: {str(e)}")
                print(f"❌ 오류가 발생했습니다: {str(e)}")


def main():
    """메인 함수"""
    try:
        setup_logging()
        logger.info("Starting Face Group Manager")
        
        manager = FaceGroupManager()
        manager.manage_groups_menu()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        logger.info("Face Group Manager finished")


if __name__ == "__main__":
    main() 