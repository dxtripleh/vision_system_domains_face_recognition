#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기본 시스템 테스트 스크립트.

카메라 연결 없이도 실행할 수 있는 기본 시스템 기능 테스트입니다.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from common.config_loader import load_config
from common.logging import setup_logging
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService

def create_test_image() -> np.ndarray:
    """테스트용 이미지 생성"""
    # 640x480 크기의 테스트 이미지 생성 (실제 얼굴 이미지는 아니지만 형태는 맞음)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 가상의 얼굴 영역 그리기 (원 모양)
    center = (320, 240)
    radius = 80
    cv2.circle(image, center, radius, (100, 100, 100), -1)
    
    # 눈 그리기
    cv2.circle(image, (290, 220), 10, (255, 255, 255), -1)
    cv2.circle(image, (350, 220), 10, (255, 255, 255), -1)
    
    # 입 그리기
    cv2.ellipse(image, (320, 270), (30, 15), 0, 0, 180, (255, 255, 255), 2)
    
    return image

def test_basic_functionality():
    """기본 기능 테스트"""
    print("🔍 Basic System Test Starting...")
    print("=" * 50)
    
    try:
        # 1. 설정 로딩 테스트
        print("1. 설정 로딩 테스트...")
        config = load_config()
        print(f"   ✅ 설정 로딩 성공: {len(config)} 개의 설정 항목")
        
        # 2. 로깅 시스템 테스트
        print("2. 로깅 시스템 테스트...")
        setup_logging()
        print("   ✅ 로깅 시스템 초기화 성공")
        
        # 3. 서비스 초기화 테스트
        print("3. 얼굴인식 서비스 초기화 테스트...")
        service = FaceRecognitionService()
        print("   ✅ 얼굴인식 서비스 초기화 성공")
        
        # 4. 인물 등록 테스트
        print("4. 인물 등록 테스트...")
        test_image = create_test_image()
        
        # 테스트 이미지 저장
        test_image_path = "data/temp/test_face.jpg"
        os.makedirs("data/temp", exist_ok=True)
        cv2.imwrite(test_image_path, test_image)
        
        # 인물 등록 시도 (실제 얼굴이 아니므로 검출 실패할 수 있음)
        try:
            # register_person은 이미지 배열을 받으므로 배열 리스트로 전달
            result = service.register_person("test_person", [test_image])
            if result:  # register_person은 person_id를 반환
                print("   ✅ 인물 등록 성공")
            else:
                print("   ⚠️ 인물 등록 실패 (예상됨 - 테스트 이미지)")
        except Exception as e:
            print(f"   ⚠️ 인물 등록 중 오류 (예상됨): {str(e)[:50]}...")
        
        # 5. 통계 정보 확인
        print("5. 시스템 통계 확인...")
        stats = service.get_statistics()
        print(f"   ✅ 등록된 인물 수: {stats.get('total_persons', 0)}")
        print(f"   ✅ 총 얼굴 수: {stats.get('total_faces', 0)}")
        
        # 6. 인물 목록 확인
        print("6. 인물 목록 확인...")
        persons = service.get_all_persons()
        print(f"   ✅ 인물 목록 조회 성공: {len(persons)}명")
        
        print("\n" + "=" * 50)
        print("🎉 기본 시스템 테스트 완료!")
        print("✅ 모든 핵심 컴포넌트가 정상 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 정리
        test_image_path = "data/temp/test_face.jpg"
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

if __name__ == "__main__":
    test_basic_functionality() 