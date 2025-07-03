#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition API 테스트 스크립트.

API 엔드포인트들을 테스트합니다.
"""

import requests
import json
import time
import cv2
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))


class APITester:
    """API 테스터 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """헬스 체크 테스트"""
        print("🔍 헬스 체크 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ 헬스 체크 성공: {data['status']}")
            return True
            
        except Exception as e:
            print(f"❌ 헬스 체크 실패: {str(e)}")
            return False
    
    def test_face_detection(self, image_path: str):
        """얼굴 검출 테스트"""
        print(f"🔍 얼굴 검출 테스트: {image_path}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/detect", files=files)
            
            response.raise_for_status()
            data = response.json()
            
            if data['success']:
                print(f"✅ 얼굴 검출 성공: {data['faces_count']}개 검출, {data['processing_time_ms']:.2f}ms")
                return True
            else:
                print(f"❌ 얼굴 검출 실패: {data}")
                return False
                
        except Exception as e:
            print(f"❌ 얼굴 검출 테스트 실패: {str(e)}")
            return False
    
    def test_face_recognition(self, image_path: str):
        """얼굴 인식 테스트"""
        print(f"🔍 얼굴 인식 테스트: {image_path}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/recognize", files=files)
            
            response.raise_for_status()
            data = response.json()
            
            if data['success']:
                print(f"✅ 얼굴 인식 성공: {len(data['faces'])}개 처리")
                for face in data['faces']:
                    person_name = face['person']['name']
                    confidence = face['confidence']
                    print(f"   - 인물: {person_name}, 신뢰도: {confidence:.3f}")
                return True
            else:
                print(f"❌ 얼굴 인식 실패: {data}")
                return False
                
        except Exception as e:
            print(f"❌ 얼굴 인식 테스트 실패: {str(e)}")
            return False
    
    def test_person_registration(self, name: str, image_paths: list):
        """인물 등록 테스트"""
        print(f"🔍 인물 등록 테스트: {name}")
        
        try:
            files = []
            for image_path in image_paths:
                with open(image_path, 'rb') as f:
                    files.append(('files', (Path(image_path).name, f.read(), 'image/jpeg')))
            
            data = {'name': name}
            response = self.session.post(f"{self.base_url}/register", data=data, files=files)
            response.raise_for_status()
            
            result = response.json()
            
            if result['success']:
                print(f"✅ 인물 등록 성공: ID={result['person_id']}")
                return result['person_id']
            else:
                print(f"❌ 인물 등록 실패: {result}")
                return None
                
        except Exception as e:
            print(f"❌ 인물 등록 테스트 실패: {str(e)}")
            return None
    
    def test_face_verification(self, person_id: str, image_path: str):
        """얼굴 검증 테스트"""
        print(f"🔍 얼굴 검증 테스트: {person_id}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                data = {'person_id': person_id}
                response = self.session.post(f"{self.base_url}/verify", data=data, files=files)
            
            response.raise_for_status()
            result = response.json()
            
            if result['success']:
                verified = result['verified']
                similarity = result['similarity']
                print(f"✅ 얼굴 검증 완료: 검증={verified}, 유사도={similarity:.3f}")
                return verified
            else:
                print(f"❌ 얼굴 검증 실패: {result}")
                return False
                
        except Exception as e:
            print(f"❌ 얼굴 검증 테스트 실패: {str(e)}")
            return False
    
    def test_statistics(self):
        """통계 조회 테스트"""
        print("🔍 통계 조회 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            
            data = response.json()
            print("✅ 통계 조회 성공")
            print(f"   - API 버전: {data['api_info']['version']}")
            print(f"   - 검출 모델: {data['detection_service']['model_name']}")
            print(f"   - 인식 임계값: {data['recognition_service']['similarity_threshold']}")
            return True
            
        except Exception as e:
            print(f"❌ 통계 조회 실패: {str(e)}")
            return False
    
    def create_test_image(self, output_path: str = "data/temp/test_image.jpg"):
        """테스트용 이미지 생성"""
        # 간단한 테스트 이미지 생성 (얼굴 모양)
        image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # 얼굴 윤곽 (원)
        cv2.circle(image, (150, 150), 80, (200, 200, 200), -1)
        
        # 눈
        cv2.circle(image, (120, 120), 10, (0, 0, 0), -1)
        cv2.circle(image, (180, 120), 10, (0, 0, 0), -1)
        
        # 코
        cv2.circle(image, (150, 150), 5, (100, 100, 100), -1)
        
        # 입
        cv2.ellipse(image, (150, 180), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        
        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        
        return output_path


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🧪 Face Recognition API 테스트")
    print("=" * 60)
    
    # API 테스터 생성
    tester = APITester()
    
    # 테스트용 이미지 생성
    test_image = tester.create_test_image()
    print(f"📸 테스트 이미지 생성됨: {test_image}")
    
    # 테스트 실행
    results = []
    
    # 1. 헬스 체크
    results.append(("헬스 체크", tester.test_health_check()))
    
    # 2. 얼굴 검출
    results.append(("얼굴 검출", tester.test_face_detection(test_image)))
    
    # 3. 얼굴 인식
    results.append(("얼굴 인식", tester.test_face_recognition(test_image)))
    
    # 4. 인물 등록
    person_id = tester.test_person_registration("TestPerson", [test_image])
    results.append(("인물 등록", person_id is not None))
    
    # 5. 얼굴 검증 (등록된 인물이 있는 경우)
    if person_id:
        results.append(("얼굴 검증", tester.test_face_verification(person_id, test_image)))
    
    # 6. 통계 조회
    results.append(("통계 조회", tester.test_statistics()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"총 테스트: {total}, 성공: {passed}, 실패: {total - passed}")
    
    if passed == total:
        print("🎉 모든 테스트가 성공했습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 