#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated System Test.

통합 시스템 테스트를 수행하는 스크립트입니다.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from common.config_loader import load_config
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService

logger = get_logger(__name__)


class IntegratedSystemTester:
    """통합 시스템 테스터"""
    
    def __init__(self, config_path: str = None):
        """초기화"""
        self.config = load_config(config_path) if config_path else {}
        self.test_results = []
        self.temp_dir = None
        
        # 서비스 초기화
        self.detection_service = FaceDetectionService(config=self.config.get('detection', {}))
        self.recognition_service = FaceRecognitionService(config=self.config.get('recognition', {}))
        
        logger.info("통합 시스템 테스터 초기화 완료")
    
    def setUp(self):
        """테스트 환경 설정"""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp(prefix="face_recognition_test_")
        
        # 테스트용 이미지 생성
        self._create_test_images()
        
        logger.info(f"테스트 환경 설정 완료 - 임시 디렉토리: {self.temp_dir}")
    
    def tearDown(self):
        """테스트 환경 정리"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("테스트 환경 정리 완료")
    
    def _create_test_images(self):
        """테스트용 이미지 생성"""
        import cv2
        import numpy as np
        
        # 가상의 얼굴 이미지 생성 (실제로는 간단한 패턴)
        for i in range(3):
            # 112x112 크기의 가상 얼굴 이미지
            test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # 얼굴처럼 보이는 패턴 추가
            cv2.circle(test_image, (56, 56), 40, (200, 200, 200), -1)  # 얼굴
            cv2.circle(test_image, (40, 40), 8, (50, 50, 50), -1)      # 왼쪽 눈
            cv2.circle(test_image, (72, 40), 8, (50, 50, 50), -1)      # 오른쪽 눈
            cv2.circle(test_image, (56, 56), 5, (100, 100, 100), -1)   # 코
            cv2.ellipse(test_image, (56, 72), (15, 8), 0, 0, 180, (100, 100, 100), 2)  # 입
            
            # 개인별 특징 추가
            if i == 0:  # 사람 A
                cv2.rectangle(test_image, (35, 20), (77, 25), (0, 0, 0), -1)  # 눈썹
            elif i == 1:  # 사람 B
                cv2.circle(test_image, (30, 70), 3, (0, 0, 0), -1)  # 점
            # 사람 C는 기본 패턴
            
            image_path = os.path.join(self.temp_dir, f"test_face_{i}.jpg")
            cv2.imwrite(image_path, test_image)
        
        logger.info("테스트용 이미지 3개 생성 완료")
    
    def test_value_objects(self) -> Dict[str, Any]:
        """Value Objects 테스트"""
        logger.info("🧪 Value Objects 테스트 시작")
        
        test_result = {
            'test_name': 'Value Objects Test',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            # BoundingBox 테스트
            from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
            bbox = BoundingBox(x=10, y=20, width=100, height=150)
            assert bbox.area == 15000
            assert bbox.center == (60.0, 95.0)
            test_result['details'].append("BoundingBox 생성 및 속성 계산: PASS")
            
            # ConfidenceScore 테스트
            from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
            conf = ConfidenceScore(0.85)
            assert conf.is_high == True
            assert conf.percentage == 85.0
            test_result['details'].append("ConfidenceScore 생성 및 속성 계산: PASS")
            
            # FaceEmbedding 테스트
            from domains.face_recognition.core.value_objects.face_embedding import FaceEmbedding
            import numpy as np
            
            vector = np.random.randn(512).astype(np.float32)
            embedding = FaceEmbedding(vector=vector, model_name="test")
            assert embedding.dimension == 512
            test_result['details'].append("FaceEmbedding 생성 및 속성 계산: PASS")
            
            # Point 테스트
            from domains.face_recognition.core.value_objects.point import Point
            point1 = Point(x=0, y=0)
            point2 = Point(x=3, y=4)
            distance = point1.distance_to(point2)
            assert abs(distance - 5.0) < 0.001
            test_result['details'].append("Point 생성 및 거리 계산: PASS")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            logger.error(f"Value Objects 테스트 실패: {str(e)}")
        
        logger.info(f"✅ Value Objects 테스트 완료 - 상태: {test_result['status']}")
        return test_result
    
    def test_repositories(self) -> Dict[str, Any]:
        """Repository 테스트"""
        logger.info("🧪 Repository 테스트 시작")
        
        test_result = {
            'test_name': 'Repository Test',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            from domains.face_recognition.core.repositories.person_repository import PersonRepository
            from domains.face_recognition.core.repositories.face_repository import FaceRepository
            from domains.face_recognition.core.entities.person import Person
            from domains.face_recognition.core.entities.face import Face
            from domains.face_recognition.core.value_objects.face_embedding import FaceEmbedding
            from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
            from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
            import numpy as np
            
            # 임시 저장소 경로
            temp_storage = os.path.join(self.temp_dir, "storage")
            os.makedirs(temp_storage, exist_ok=True)
            
            # Repository 초기화
            person_repo = PersonRepository(storage_path=os.path.join(temp_storage, "persons"))
            face_repo = FaceRepository(storage_path=os.path.join(temp_storage, "faces"))
            
            # 테스트 데이터 생성
            vector = np.random.randn(512).astype(np.float32)
            embedding = FaceEmbedding(vector=vector, model_name="test")
            
            face = Face(
                face_id="test_face_1",
                person_id="test_person_1",
                embedding=embedding,
                bbox=BoundingBox(x=0, y=0, width=100, height=100),
                confidence=ConfidenceScore(0.9),
                timestamp=time.time(),
                quality_score=0.8
            )
            
            person = Person(
                person_id="test_person_1",
                name="테스트 사용자",
                faces=[face],
                metadata={"test": True},
                created_at=time.time()
            )
            
            # 저장 테스트
            assert face_repo.save(face) == True
            assert person_repo.save(person) == True
            test_result['details'].append("데이터 저장: PASS")
            
            # 조회 테스트
            retrieved_face = face_repo.find_by_id("test_face_1")
            retrieved_person = person_repo.find_by_id("test_person_1")
            
            assert retrieved_face is not None
            assert retrieved_person is not None
            assert retrieved_person.name == "테스트 사용자"
            test_result['details'].append("데이터 조회: PASS")
            
            # 삭제 테스트
            assert face_repo.delete("test_face_1") == True
            assert person_repo.delete("test_person_1") == True
            test_result['details'].append("데이터 삭제: PASS")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            logger.error(f"Repository 테스트 실패: {str(e)}")
        
        logger.info(f"✅ Repository 테스트 완료 - 상태: {test_result['status']}")
        return test_result
    
    def test_face_detection_service(self) -> Dict[str, Any]:
        """얼굴 검출 서비스 테스트"""
        logger.info("🧪 얼굴 검출 서비스 테스트 시작")
        
        test_result = {
            'test_name': 'Face Detection Service Test',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            import cv2
            
            # 테스트 이미지 로드
            test_image_path = os.path.join(self.temp_dir, "test_face_0.jpg")
            image = cv2.imread(test_image_path)
            
            if image is None:
                raise ValueError(f"테스트 이미지를 로드할 수 없습니다: {test_image_path}")
            
            # 얼굴 검출 수행
            detection_result = self.detection_service.detect_faces(image)
            
            # 결과 검증
            assert detection_result is not None
            assert hasattr(detection_result, 'faces')
            assert hasattr(detection_result, 'processing_time_ms')
            test_result['details'].append(f"얼굴 검출 수행: PASS (검출된 얼굴 수: {len(detection_result.faces)})")
            
            # 처리 시간 검증
            assert detection_result.processing_time_ms >= 0
            test_result['details'].append(f"처리 시간 측정: PASS ({detection_result.processing_time_ms:.2f}ms)")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            logger.error(f"얼굴 검출 서비스 테스트 실패: {str(e)}")
        
        logger.info(f"✅ 얼굴 검출 서비스 테스트 완료 - 상태: {test_result['status']}")
        return test_result
    
    def test_face_recognition_service(self) -> Dict[str, Any]:
        """얼굴 인식 서비스 테스트"""
        logger.info("🧪 얼굴 인식 서비스 테스트 시작")
        
        test_result = {
            'test_name': 'Face Recognition Service Test',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            import cv2
            
            # 테스트 이미지들 로드
            test_images = []
            for i in range(3):
                image_path = os.path.join(self.temp_dir, f"test_face_{i}.jpg")
                image = cv2.imread(image_path)
                if image is not None:
                    test_images.append(image)
            
            if not test_images:
                raise ValueError("테스트 이미지를 로드할 수 없습니다")
            
            # 임베딩 추출 테스트
            embedding = self.recognition_service.extract_embedding(test_images[0])
            assert embedding is not None
            assert embedding.dimension > 0
            test_result['details'].append(f"임베딩 추출: PASS (차원: {embedding.dimension})")
            
            # 인물 등록 테스트
            person_id = self.recognition_service.register_person(
                name="테스트 사용자 A",
                face_images=[test_images[0]]
            )
            assert person_id is not None
            test_result['details'].append("인물 등록: PASS")
            
            # 인물 조회 테스트
            person = self.recognition_service.get_person_by_id(person_id)
            assert person is not None
            assert person.name == "테스트 사용자 A"
            test_result['details'].append("인물 조회: PASS")
            
            # 통계 조회 테스트
            stats = self.recognition_service.get_statistics()
            assert isinstance(stats, dict)
            assert 'total_persons' in stats
            test_result['details'].append("통계 조회: PASS")
            
            # 인물 삭제 테스트
            deleted = self.recognition_service.remove_person(person_id)
            assert deleted == True
            test_result['details'].append("인물 삭제: PASS")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            logger.error(f"얼굴 인식 서비스 테스트 실패: {str(e)}")
        
        logger.info(f"✅ 얼굴 인식 서비스 테스트 완료 - 상태: {test_result['status']}")
        return test_result
    
    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """종단간 워크플로우 테스트"""
        logger.info("🧪 종단간 워크플로우 테스트 시작")
        
        test_result = {
            'test_name': 'End-to-End Workflow Test',
            'status': 'PASS',
            'details': [],
            'errors': []
        }
        
        try:
            import cv2
            from domains.face_recognition.core.entities.face import Face
            from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
            from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
            
            # 1. 이미지 로드
            test_image = cv2.imread(os.path.join(self.temp_dir, "test_face_0.jpg"))
            registration_image = cv2.imread(os.path.join(self.temp_dir, "test_face_1.jpg"))
            
            # 2. 인물 등록
            person_id = self.recognition_service.register_person(
                name="워크플로우 테스트 사용자",
                face_images=[registration_image],
                metadata={"test_type": "end_to_end"}
            )
            test_result['details'].append("1. 인물 등록 완료")
            
            # 3. 얼굴 검출
            detection_result = self.detection_service.detect_faces(test_image)
            if detection_result.faces:
                detected_face = detection_result.faces[0]
                test_result['details'].append("2. 얼굴 검출 완료")
            else:
                # 검출되지 않은 경우 가상의 얼굴 생성
                detected_face = Face(
                    face_id="virtual_face",
                    person_id="",
                    embedding=None,
                    bbox=BoundingBox(x=0, y=0, width=test_image.shape[1], height=test_image.shape[0]),
                    confidence=ConfidenceScore(0.5),
                    timestamp=time.time(),
                    quality_score=0.5
                )
                test_result['details'].append("2. 가상 얼굴 생성 (검출 실패)")
            
            # 4. 임베딩 추출
            embedding = self.recognition_service.extract_embedding(test_image)
            detected_face.embedding = embedding
            test_result['details'].append("3. 임베딩 추출 완료")
            
            # 5. 얼굴 식별
            identified_person = self.recognition_service.identify_face(detected_face)
            if identified_person:
                test_result['details'].append(f"4. 얼굴 식별 완료: {identified_person.name}")
            else:
                test_result['details'].append("4. 얼굴 식별 결과: 미등록 인물")
            
            # 6. 얼굴 검증
            is_match, similarity = self.recognition_service.verify_face(detected_face, person_id)
            test_result['details'].append(f"5. 얼굴 검증 완료: {'일치' if is_match else '불일치'} (유사도: {similarity:.3f})")
            
            # 7. 정리
            self.recognition_service.remove_person(person_id)
            test_result['details'].append("6. 테스트 데이터 정리 완료")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['errors'].append(str(e))
            logger.error(f"종단간 워크플로우 테스트 실패: {str(e)}")
        
        logger.info(f"✅ 종단간 워크플로우 테스트 완료 - 상태: {test_result['status']}")
        return test_result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """모든 테스트 실행"""
        logger.info("🚀 통합 시스템 테스트 시작")
        
        self.setUp()
        
        try:
            # 모든 테스트 실행
            self.test_results = [
                self.test_value_objects(),
                self.test_repositories(),
                self.test_face_detection_service(),
                self.test_face_recognition_service(),
                self.test_end_to_end_workflow()
            ]
            
        finally:
            self.tearDown()
        
        # 결과 요약
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = total_tests - passed_tests
        
        logger.info("="*60)
        logger.info("📊 테스트 결과 요약")
        logger.info("="*60)
        logger.info(f"총 테스트 수: {total_tests}")
        logger.info(f"성공: {passed_tests}")
        logger.info(f"실패: {failed_tests}")
        logger.info(f"성공률: {(passed_tests/total_tests)*100:.1f}%")
        
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            logger.info(f"{status_icon} {result['test_name']}: {result['status']}")
            
            for detail in result['details']:
                logger.info(f"   - {detail}")
            
            for error in result['errors']:
                logger.error(f"   ❌ {error}")
        
        logger.info("="*60)
        
        return self.test_results


def main():
    """메인 함수"""
    setup_logging()
    
    logger.info("🧪 통합 시스템 테스트 시작")
    
    try:
        # 테스터 초기화
        config_path = "config/face_recognition_config.yaml"
        tester = IntegratedSystemTester(config_path)
        
        # 모든 테스트 실행
        results = tester.run_all_tests()
        
        # 종료 코드 결정
        failed_tests = [r for r in results if r['status'] == 'FAIL']
        exit_code = 1 if failed_tests else 0
        
        if exit_code == 0:
            logger.info("🎉 모든 테스트가 성공했습니다!")
        else:
            logger.error(f"💥 {len(failed_tests)}개 테스트가 실패했습니다.")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 