#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
얼굴 유사도 디버깅 도구

두 얼굴 이미지의 유사도를 상세히 분석하여 그룹핑 문제를 진단합니다.

사용법:
    python debug_face_similarity.py face1.jpg face2.jpg
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger

# CrossCheckRecognizer 클래스를 run_unified_ai_grouping_processor.py에서 가져오기
try:
    from domains.face_recognition.runners.data_collection.run_unified_ai_grouping_processor import CrossCheckRecognizer, FacePreprocessor
except ImportError as e:
    print(f"❌ CrossCheckRecognizer를 가져올 수 없습니다: {e}")
    sys.exit(1)

def compare_two_faces(face_path1: str, face_path2: str):
    """두 얼굴 이미지의 유사도를 상세 분석"""
    
    # 로깅 설정
    setup_logging()
    logger = get_logger(__name__)
    
    print(f"\n🔍 얼굴 유사도 분석")
    print(f"=" * 60)
    print(f"얼굴 1: {face_path1}")
    print(f"얼굴 2: {face_path2}")
    print(f"=" * 60)
    
    # 파일 존재 확인
    if not Path(face_path1).exists():
        print(f"❌ 파일이 존재하지 않습니다: {face_path1}")
        return
    
    if not Path(face_path2).exists():
        print(f"❌ 파일이 존재하지 않습니다: {face_path2}")
        return
    
    # 이미지 로드
    face1 = cv2.imread(face_path1)
    face2 = cv2.imread(face_path2)
    
    if face1 is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {face_path1}")
        return
    
    if face2 is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {face_path2}")
        return
    
    print(f"✅ 이미지 로드 완료")
    print(f"   얼굴 1 크기: {face1.shape}")
    print(f"   얼굴 2 크기: {face2.shape}")
    
    # 전처리기 및 인식기 초기화
    print(f"\n🤖 AI 모델 초기화 중...")
    
    try:
        preprocessor = FacePreprocessor()
        recognizer = CrossCheckRecognizer()
        
        print(f"✅ 모델 초기화 완료")
        print(f"   사용 가능한 모델: {len(recognizer.models)}개")
        for model_name, model_info in recognizer.models.items():
            print(f"   - {model_info['name']} (가중치: {model_info['weight']})")
        
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return
    
    # 전처리
    print(f"\n🔧 얼굴 전처리 중...")
    
    processed_face1 = preprocessor.preprocess_face(face1)
    processed_face2 = preprocessor.preprocess_face(face2)
    
    if processed_face1 is None:
        print(f"⚠️ 얼굴 1 전처리 실패 - 원본 사용")
        processed_face1 = face1
    
    if processed_face2 is None:
        print(f"⚠️ 얼굴 2 전처리 실패 - 원본 사용")
        processed_face2 = face2
    
    print(f"✅ 전처리 완료")
    
    # 특징 추출
    print(f"\n🎯 특징 추출 중...")
    
    features1 = recognizer.extract_features(processed_face1)
    features2 = recognizer.extract_features(processed_face2)
    
    if not features1:
        print(f"❌ 얼굴 1 특징 추출 실패")
        return
    
    if not features2:
        print(f"❌ 얼굴 2 특징 추출 실패")
        return
    
    print(f"✅ 특징 추출 완료")
    print(f"   얼굴 1 특징: {list(features1.keys())}")
    print(f"   얼굴 2 특징: {list(features2.keys())}")
    
    # 유사도 계산
    print(f"\n📊 유사도 계산 중...")
    
    similarities = recognizer.calculate_similarity(features1, features2)
    
    print(f"\n🔍 모델별 유사도 분석:")
    print(f"-" * 40)
    
    for model_name, similarity in similarities.items():
        if model_name in recognizer.models:
            weight = recognizer.models[model_name]['weight']
            threshold = recognizer.models[model_name]['threshold']
            status = "✅ 같은 사람" if similarity >= threshold else "❌ 다른 사람"
            print(f"{model_name:15s}: {similarity:.3f} (임계값: {threshold}, 가중치: {weight}) {status}")
        else:
            print(f"{model_name:15s}: {similarity:.3f}")
    
    # 컨센서스 계산
    debug_info = {
        'face1': Path(face_path1).name,
        'face2': Path(face_path2).name
    }
    
    consensus_similarity = recognizer.get_consensus_similarity(similarities, debug_info)
    
    print(f"\n📊 최종 컨센서스 결과:")
    print(f"-" * 40)
    print(f"컨센서스 점수: {consensus_similarity:.3f}")
    
    # 기본 임계값으로 판정
    default_threshold = 0.75
    final_result = "✅ 같은 사람" if consensus_similarity >= default_threshold else "❌ 다른 사람"
    print(f"판정 결과 (임계값 {default_threshold}): {final_result}")
    
    print(f"\n💡 임계값 조정 가이드:")
    print(f"-" * 40)
    for threshold in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        result = "같은 사람" if consensus_similarity >= threshold else "다른 사람"
        print(f"임계값 {threshold}: {result}")
    
    # 이미지 저장 (선택적)
    save_analysis = input(f"\n💾 분석 결과를 이미지로 저장하시겠습니까? (y/N): ").lower().strip()
    
    if save_analysis in ['y', 'yes']:
        output_dir = Path("data/output/debug_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 나란히 배치한 이미지 생성
        height = max(face1.shape[0], face2.shape[0])
        
        # 이미지 크기 조정
        face1_resized = cv2.resize(face1, (200, height))
        face2_resized = cv2.resize(face2, (200, height))
        
        # 나란히 배치
        combined = np.hstack([face1_resized, face2_resized])
        
        # 텍스트 추가
        cv2.putText(combined, f"Similarity: {consensus_similarity:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"Result: {final_result}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        output_path = output_dir / f"similarity_analysis_{consensus_similarity:.3f}.jpg"
        cv2.imwrite(str(output_path), combined)
        
        print(f"✅ 분석 결과 저장됨: {output_path}")
    
    print(f"\n✅ 분석 완료!")

def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("사용법: python debug_face_similarity.py <얼굴1_경로> <얼굴2_경로>")
        print("")
        print("예시:")
        print("  python debug_face_similarity.py face1.jpg face2.jpg")
        print("  python debug_face_similarity.py \"data/domains/face_recognition/detected_faces/from_uploads/face_image01_20250630_125353_251_00_conf1.00.jpg\" \"data/domains/face_recognition/detected_faces/from_captured/face_captured_frame_20250630_110817_109_20250630_131421_836_00_conf1.00.jpg\"")
        sys.exit(1)
    
    face_path1 = sys.argv[1]
    face_path2 = sys.argv[2]
    
    compare_two_faces(face_path1, face_path2)

if __name__ == "__main__":
    main() 