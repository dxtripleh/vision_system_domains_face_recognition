#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition Demo.

간단한 얼굴인식 데모 스크립트입니다.
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from ..services.service import FaceRecognitionService

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """테스트용 이미지 생성"""
    # 랜덤 배경
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 중앙에 얼굴 모양 그리기
    center_x, center_y = width // 2, height // 2
    face_size = min(width, height) // 4
    
    # 얼굴 영역
    cv2.rectangle(image, 
                 (center_x - face_size//2, center_y - face_size//2),
                 (center_x + face_size//2, center_y + face_size//2),
                 (255, 255, 255), -1)
    
    # 눈, 코, 입
    cv2.circle(image, (center_x - 20, center_y - 10), 5, (0, 0, 0), -1)
    cv2.circle(image, (center_x + 20, center_y - 10), 5, (0, 0, 0), -1)
    cv2.circle(image, (center_x, center_y), 3, (0, 0, 0), -1)
    cv2.ellipse(image, (center_x, center_y + 15), (15, 8), 0, 0, 180, (0, 0, 0), 2)
    
    return image

def main():
    """데모 실행"""
    print("=== 얼굴인식 데모 시작 ===")
    
    try:
        # 서비스 초기화
        print("얼굴인식 서비스 초기화 중...")
        service = FaceRecognitionService()
        print("✓ 서비스 초기화 완료")
        
        # 테스트 이미지 생성
        print("테스트 이미지 생성 중...")
        test_image = create_test_image()
        print("✓ 테스트 이미지 생성 완료")
        
        # 얼굴인식 수행
        print("얼굴인식 처리 중...")
        start_time = time.time()
        result = service.process_frame(test_image)
        processing_time = time.time() - start_time
        
        # 결과 출력
        print(f"✓ 얼굴인식 완료!")
        print(f"  - 검출된 얼굴: {len(result['faces'])}개")
        print(f"  - 처리 시간: {processing_time*1000:.1f}ms")
        print(f"  - 통계: {result['stats']}")
        
        # 결과 이미지 표시 (OpenCV 창)
        if len(result['faces']) > 0:
            print("결과 이미지 표시 중... (ESC 키로 종료)")
            cv2.imshow('Face Recognition Demo', result['frame_with_results'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("=== 데모 완료 ===")
        return 0
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 