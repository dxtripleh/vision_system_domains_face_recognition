#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 전처리 및 크로스체크 시스템 테스트

이 스크립트는 얼굴 전처리 및 크로스체크 시스템의 기능을 테스트합니다.

사용법:
    python scripts/test/test_face_preprocessing.py [--test_image path/to/image.jpg]
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import time

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger

def test_face_preprocessing():
    """얼굴 전처리 테스트"""
    logger = get_logger(__name__)
    
    try:
        # FacePreprocessor 임포트
        from domains.face_recognition.runners.data_collection.run_unified_ai_grouping_processor import FacePreprocessor
        
        logger.info("얼굴 전처리 테스트 시작")
        
        # 전처리기 초기화
        preprocessor = FacePreprocessor()
        
        # 테스트 이미지 생성 (더미 얼굴 이미지)
        test_image = create_dummy_face_image()
        
        # 전처리 실행
        start_time = time.time()
        preprocessed = preprocessor.preprocess_face(test_image)
        processing_time = time.time() - start_time
        
        if preprocessed is not None:
            logger.info(f"전처리 성공: {preprocessed.shape}")
            logger.info(f"처리 시간: {processing_time:.3f}초")
            
            # 결과 저장
            output_path = project_root / 'data' / 'temp' / 'test_preprocessed.jpg'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), preprocessed)
            logger.info(f"전처리 결과 저장: {output_path}")
            
            return True
        else:
            logger.error("전처리 실패")
            return False
            
    except ImportError as e:
        logger.error(f"모듈 임포트 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"전처리 테스트 오류: {e}")
        return False

def test_crosscheck_recognition():
    """크로스체크 인식 테스트"""
    logger = get_logger(__name__)
    
    try:
        # CrossCheckRecognizer 임포트
        from domains.face_recognition.runners.data_collection.run_unified_ai_grouping_processor import CrossCheckRecognizer
        
        logger.info("크로스체크 인식 테스트 시작")
        
        # 인식기 초기화
        recognizer = CrossCheckRecognizer()
        
        # 테스트 이미지들 생성
        face1 = create_dummy_face_image(person_id=1)
        face2 = create_dummy_face_image(person_id=1)  # 같은 사람
        face3 = create_dummy_face_image(person_id=2)  # 다른 사람
        
        # 특징 추출
        start_time = time.time()
        features1 = recognizer.extract_features(face1)
        features2 = recognizer.extract_features(face2)
        features3 = recognizer.extract_features(face3)
        extraction_time = time.time() - start_time
        
        logger.info(f"특징 추출 완료: {extraction_time:.3f}초")
        logger.info(f"사용 가능한 모델: {list(features1.keys())}")
        
        # 유사도 계산
        start_time = time.time()
        similarity_same = recognizer.calculate_similarity(features1, features2)
        similarity_diff = recognizer.calculate_similarity(features1, features3)
        similarity_time = time.time() - start_time
        
        # 합의 유사도 계산
        consensus_same = recognizer.get_consensus_similarity(similarity_same)
        consensus_diff = recognizer.get_consensus_similarity(similarity_diff)
        
        logger.info(f"유사도 계산 완료: {similarity_time:.3f}초")
        logger.info(f"같은 사람 유사도: {consensus_same:.3f}")
        logger.info(f"다른 사람 유사도: {consensus_diff:.3f}")
        
        # 상세 모델별 유사도 출력
        logger.info("모델별 유사도 (같은 사람):")
        for model_name, sim in similarity_same.items():
            logger.info(f"  {model_name}: {sim:.3f}")
        
        logger.info("모델별 유사도 (다른 사람):")
        for model_name, sim in similarity_diff.items():
            logger.info(f"  {model_name}: {sim:.3f}")
        
        return True
        
    except ImportError as e:
        logger.error(f"모듈 임포트 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"크로스체크 테스트 오류: {e}")
        return False

def test_integration():
    """통합 테스트"""
    logger = get_logger(__name__)
    
    try:
        from domains.face_recognition.runners.data_collection.run_unified_ai_grouping_processor import (
            FacePreprocessor, CrossCheckRecognizer
        )
        
        logger.info("통합 테스트 시작")
        
        # 컴포넌트 초기화
        preprocessor = FacePreprocessor()
        recognizer = CrossCheckRecognizer()
        
        # 테스트 데이터 생성
        test_faces = []
        for i in range(3):
            face = create_dummy_face_image(person_id=i)
            test_faces.append(face)
        
        # 전체 파이프라인 테스트
        start_time = time.time()
        
        features_list = []
        for i, face in enumerate(test_faces):
            # 1. 전처리
            preprocessed = preprocessor.preprocess_face(face)
            if preprocessed is None:
                logger.warning(f"얼굴 {i} 전처리 실패")
                continue
            
            # 2. 특징 추출
            features = recognizer.extract_features(preprocessed)
            if features:
                features_list.append(features)
                logger.info(f"얼굴 {i} 처리 완료: {len(features)}개 모델")
            else:
                logger.warning(f"얼굴 {i} 특징 추출 실패")
        
        total_time = time.time() - start_time
        
        logger.info(f"통합 테스트 완료: {total_time:.3f}초")
        logger.info(f"성공적으로 처리된 얼굴: {len(features_list)}/{len(test_faces)}")
        
        # 유사도 테스트
        if len(features_list) >= 2:
            similarity = recognizer.calculate_similarity(features_list[0], features_list[1])
            consensus = recognizer.get_consensus_similarity(similarity)
            logger.info(f"샘플 유사도: {consensus:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"통합 테스트 오류: {e}")
        return False

def create_dummy_face_image(person_id: int = 1) -> np.ndarray:
    """테스트용 더미 얼굴 이미지 생성"""
    # 200x200 크기의 더미 얼굴 이미지
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 배경색 (피부톤)
    image[:, :] = [255, 220, 177]
    
    # 얼굴 윤곽 (타원)
    cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, (200, 150, 100), -1)
    
    # 눈 (타원)
    cv2.ellipse(image, (80, 80), (8, 5), 0, 0, 360, (0, 0, 0), -1)
    cv2.ellipse(image, (120, 80), (8, 5), 0, 0, 360, (0, 0, 0), -1)
    
    # 코 (삼각형)
    pts = np.array([[100, 90], [95, 110], [105, 110]], np.int32)
    cv2.fillPoly(image, [pts], (150, 100, 50))
    
    # 입 (타원)
    cv2.ellipse(image, (100, 130), (15, 8), 0, 0, 180, (100, 50, 50), -1)
    
    # 사람별로 약간 다른 특성 추가
    if person_id == 2:
        # 다른 사람은 다른 색상
        image = cv2.addWeighted(image, 0.8, np.full_like(image, [50, 0, 0]), 0.2, 0)
    
    return image

def test_with_real_image(image_path: str):
    """실제 이미지로 테스트"""
    logger = get_logger(__name__)
    
    if not os.path.exists(image_path):
        logger.error(f"이미지 파일 없음: {image_path}")
        return False
    
    try:
        from domains.face_recognition.runners.data_collection.run_unified_ai_grouping_processor import (
            FacePreprocessor, CrossCheckRecognizer
        )
        
        logger.info(f"실제 이미지 테스트: {image_path}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error("이미지 로드 실패")
            return False
        
        logger.info(f"이미지 크기: {image.shape}")
        
        # 전처리기 및 인식기 초기화
        preprocessor = FacePreprocessor()
        recognizer = CrossCheckRecognizer()
        
        # 전처리
        start_time = time.time()
        preprocessed = preprocessor.preprocess_face(image)
        preprocess_time = time.time() - start_time
        
        if preprocessed is not None:
            logger.info(f"전처리 성공: {preprocessed.shape}")
            logger.info(f"전처리 시간: {preprocess_time:.3f}초")
            
            # 특징 추출
            start_time = time.time()
            features = recognizer.extract_features(preprocessed)
            extraction_time = time.time() - start_time
            
            if features:
                logger.info(f"특징 추출 성공: {len(features)}개 모델")
                logger.info(f"추출 시간: {extraction_time:.3f}초")
                
                for model_name, feature in features.items():
                    logger.info(f"  {model_name}: {feature.shape}")
                
                # 결과 저장
                output_path = project_root / 'data' / 'temp' / f'test_real_preprocessed_{Path(image_path).stem}.jpg'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), preprocessed)
                logger.info(f"전처리 결과 저장: {output_path}")
                
                return True
            else:
                logger.error("특징 추출 실패")
                return False
        else:
            logger.error("전처리 실패")
            return False
            
    except Exception as e:
        logger.error(f"실제 이미지 테스트 오류: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="얼굴 전처리 및 크로스체크 시스템 테스트")
    parser.add_argument("--test_image", type=str, help="테스트할 실제 이미지 경로")
    parser.add_argument("--skip_preprocessing", action="store_true", help="전처리 테스트 건너뛰기")
    parser.add_argument("--skip_crosscheck", action="store_true", help="크로스체크 테스트 건너뛰기")
    parser.add_argument("--skip_integration", action="store_true", help="통합 테스트 건너뛰기")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("얼굴 전처리 및 크로스체크 시스템 테스트 시작")
    
    results = {}
    
    # 1. 전처리 테스트
    if not args.skip_preprocessing:
        logger.info("=" * 50)
        logger.info("1. 얼굴 전처리 테스트")
        results['preprocessing'] = test_face_preprocessing()
    
    # 2. 크로스체크 테스트
    if not args.skip_crosscheck:
        logger.info("=" * 50)
        logger.info("2. 크로스체크 인식 테스트")
        results['crosscheck'] = test_crosscheck_recognition()
    
    # 3. 통합 테스트
    if not args.skip_integration:
        logger.info("=" * 50)
        logger.info("3. 통합 테스트")
        results['integration'] = test_integration()
    
    # 4. 실제 이미지 테스트
    if args.test_image:
        logger.info("=" * 50)
        logger.info("4. 실제 이미지 테스트")
        results['real_image'] = test_with_real_image(args.test_image)
    
    # 결과 요약
    logger.info("=" * 50)
    logger.info("테스트 결과 요약:")
    
    for test_name, result in results.items():
        status = "✓ 성공" if result else "✗ 실패"
        logger.info(f"  {test_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        logger.info("모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        return 0
    else:
        logger.warning("일부 테스트 실패. 로그를 확인하여 문제를 해결하세요.")
        return 1

if __name__ == "__main__":
    exit(main()) 