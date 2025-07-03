#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition System Runner.

얼굴 인식 시스템의 메인 실행 스크립트입니다.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from common.config_loader import load_config
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.infrastructure.models.retinaface_detector import RetinaFaceDetector
from domains.face_recognition.infrastructure.models.arcface_recognizer import ArcFaceRecognizer

logger = get_logger(__name__)


def create_face_detection_service(config: dict) -> FaceDetectionService:
    """얼굴 검출 서비스 생성"""
    detection_config = config.get('detection', {})
    
    try:
        # RetinaFace 검출기 초기화
        detector = RetinaFaceDetector(
            model_path=detection_config.get('model_path', 'models/weights/face_detection_retinaface_mnet025_20250628.onnx'),
            confidence_threshold=detection_config.get('confidence_threshold', 0.5),
            nms_threshold=detection_config.get('nms_threshold', 0.4),
            use_gpu=detection_config.get('use_gpu', True)
        )
        
        if detector.load_model():
            logger.info("RetinaFace detector loaded successfully")
        else:
            logger.warning("Failed to load RetinaFace detector, using fallback")
            detector = None
        
        return FaceDetectionService(detector=detector, config=detection_config)
        
    except Exception as e:
        logger.error(f"Error creating face detection service: {str(e)}")
        return FaceDetectionService(config=detection_config)


def create_face_recognition_service(config: dict) -> FaceRecognitionService:
    """얼굴 인식 서비스 생성"""
    recognition_config = config.get('recognition', {})
    
    try:
        # ArcFace 인식기 초기화
        recognizer = ArcFaceRecognizer(
            model_path=recognition_config.get('model_path', 'models/weights/face_recognition_arcface_r50_20250628.onnx'),
            input_size=recognition_config.get('input_size', (112, 112)),
            use_gpu=recognition_config.get('use_gpu', True)
        )
        
        if recognizer.load_model():
            logger.info("ArcFace recognizer loaded successfully")
        else:
            logger.warning("Failed to load ArcFace recognizer, using fallback")
            recognizer = None
        
        return FaceRecognitionService(recognizer=recognizer, config=recognition_config)
        
    except Exception as e:
        logger.error(f"Error creating face recognition service: {str(e)}")
        return FaceRecognitionService(config=recognition_config)


def run_interactive_demo(detection_service: FaceDetectionService, 
                        recognition_service: FaceRecognitionService):
    """대화형 데모 실행"""
    print("\n" + "="*50)
    print("🎯 얼굴 인식 시스템 대화형 데모")
    print("="*50)
    
    while True:
        print("\n📋 사용 가능한 명령:")
        print("1. register - 새로운 인물 등록")
        print("2. identify - 얼굴 식별")
        print("3. verify - 얼굴 검증")
        print("4. list - 등록된 인물 목록")
        print("5. stats - 시스템 통계")
        print("6. exit - 종료")
        
        command = input("\n💡 명령을 선택하세요: ").strip().lower()
        
        if command == "exit" or command == "6":
            print("👋 시스템을 종료합니다.")
            break
        elif command == "register" or command == "1":
            handle_register_command(recognition_service)
        elif command == "identify" or command == "2":
            handle_identify_command(detection_service, recognition_service)
        elif command == "verify" or command == "3":
            handle_verify_command(recognition_service)
        elif command == "list" or command == "4":
            handle_list_command(recognition_service)
        elif command == "stats" or command == "5":
            handle_stats_command(recognition_service)
        else:
            print("❌ 알 수 없는 명령입니다.")


def handle_register_command(recognition_service: FaceRecognitionService):
    """인물 등록 명령 처리"""
    try:
        name = input("📝 등록할 인물의 이름을 입력하세요: ").strip()
        if not name:
            print("❌ 이름을 입력해주세요.")
            return
        
        image_path = input("📷 얼굴 이미지 경로를 입력하세요: ").strip()
        if not os.path.exists(image_path):
            print("❌ 이미지 파일을 찾을 수 없습니다.")
            return
        
        # 이미지 로드
        import cv2
        face_image = cv2.imread(image_path)
        if face_image is None:
            print("❌ 이미지를 읽을 수 없습니다.")
            return
        
        # 인물 등록
        print("⏳ 인물을 등록하는 중...")
        person_id = recognition_service.register_person(name, [face_image])
        print(f"✅ 인물 등록 완료! ID: {person_id}")
        
    except Exception as e:
        print(f"❌ 인물 등록 중 오류 발생: {str(e)}")


def handle_identify_command(detection_service: FaceDetectionService,
                          recognition_service: FaceRecognitionService):
    """얼굴 식별 명령 처리"""
    try:
        image_path = input("📷 식별할 얼굴 이미지 경로를 입력하세요: ").strip()
        if not os.path.exists(image_path):
            print("❌ 이미지 파일을 찾을 수 없습니다.")
            return
        
        # 이미지 로드
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            print("❌ 이미지를 읽을 수 없습니다.")
            return
        
        print("⏳ 얼굴을 검출하는 중...")
        
        # 얼굴 검출
        detection_result = detection_service.detect_faces(image)
        if not detection_result.faces:
            print("❌ 얼굴이 검출되지 않았습니다.")
            return
        
        print(f"✅ {len(detection_result.faces)}개의 얼굴이 검출되었습니다.")
        
        # 첫 번째 얼굴 식별
        face = detection_result.faces[0]
        
        # 얼굴 영역 추출
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
        face_image = image[y:y+h, x:x+w]
        
        # 임베딩 추출
        print("⏳ 얼굴 특징을 추출하는 중...")
        embedding = recognition_service.extract_embedding(face_image)
        face.embedding = embedding
        
        # 얼굴 식별
        print("⏳ 인물을 식별하는 중...")
        identified_person = recognition_service.identify_face(face)
        
        if identified_person:
            print(f"✅ 식별 완료: {identified_person.name}")
        else:
            print("❌ 등록된 인물과 일치하지 않습니다.")
        
    except Exception as e:
        print(f"❌ 얼굴 식별 중 오류 발생: {str(e)}")


def handle_verify_command(recognition_service: FaceRecognitionService):
    """얼굴 검증 명령 처리"""
    try:
        person_id = input("👤 검증할 인물 ID를 입력하세요: ").strip()
        if not person_id:
            print("❌ 인물 ID를 입력해주세요.")
            return
        
        image_path = input("📷 검증할 얼굴 이미지 경로를 입력하세요: ").strip()
        if not os.path.exists(image_path):
            print("❌ 이미지 파일을 찾을 수 없습니다.")
            return
        
        # 이미지 로드 및 임베딩 추출
        import cv2
        from domains.face_recognition.core.entities.face import Face
        from domains.face_recognition.core.value_objects.bounding_box import BoundingBox
        from domains.face_recognition.core.value_objects.confidence_score import ConfidenceScore
        
        face_image = cv2.imread(image_path)
        if face_image is None:
            print("❌ 이미지를 읽을 수 없습니다.")
            return
        
        print("⏳ 얼굴 특징을 추출하는 중...")
        embedding = recognition_service.extract_embedding(face_image)
        
        # Face 엔티티 생성
        face = Face(
            face_id="temp_verify",
            person_id="",
            embedding=embedding,
            bbox=BoundingBox(x=0, y=0, width=face_image.shape[1], height=face_image.shape[0]),
            confidence=ConfidenceScore(1.0),
            timestamp=0,
            quality_score=1.0
        )
        
        # 얼굴 검증
        print("⏳ 얼굴을 검증하는 중...")
        is_match, similarity = recognition_service.verify_face(face, person_id)
        
        if is_match:
            print(f"✅ 검증 성공! 유사도: {similarity:.3f}")
        else:
            print(f"❌ 검증 실패. 유사도: {similarity:.3f}")
        
    except Exception as e:
        print(f"❌ 얼굴 검증 중 오류 발생: {str(e)}")


def handle_list_command(recognition_service: FaceRecognitionService):
    """인물 목록 명령 처리"""
    try:
        print("⏳ 등록된 인물 목록을 조회하는 중...")
        persons = recognition_service.get_all_persons()
        
        if not persons:
            print("📭 등록된 인물이 없습니다.")
            return
        
        print(f"\n👥 등록된 인물 목록 ({len(persons)}명):")
        print("-" * 50)
        
        for i, person in enumerate(persons, 1):
            print(f"{i}. 이름: {person.name}")
            print(f"   ID: {person.person_id}")
            print(f"   얼굴 수: {len(person.faces)}")
            print(f"   등록일: {person.created_at}")
            print()
        
    except Exception as e:
        print(f"❌ 인물 목록 조회 중 오류 발생: {str(e)}")


def handle_stats_command(recognition_service: FaceRecognitionService):
    """통계 명령 처리"""
    try:
        print("⏳ 시스템 통계를 조회하는 중...")
        stats = recognition_service.get_statistics()
        
        print("\n📊 시스템 통계:")
        print("-" * 30)
        print(f"등록된 인물 수: {stats.get('total_persons', 0)}")
        print(f"총 얼굴 수: {stats.get('total_faces', 0)}")
        print(f"인물당 평균 얼굴 수: {stats.get('average_faces_per_person', 0):.1f}")
        print(f"유사도 임계값: {stats.get('similarity_threshold', 0):.2f}")
        print(f"임베딩 차원: {stats.get('embedding_dimension', 0)}")
        print(f"AI 모델 사용 가능: {'예' if stats.get('recognizer_available', False) else '아니오'}")
        
    except Exception as e:
        print(f"❌ 통계 조회 중 오류 발생: {str(e)}")


def run_batch_processing(detection_service: FaceDetectionService,
                        recognition_service: FaceRecognitionService,
                        input_dir: str, output_dir: str):
    """배치 처리 실행"""
    logger.info(f"배치 처리 시작 - 입력: {input_dir}, 출력: {output_dir}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"입력 디렉토리에 이미지 파일이 없습니다: {input_dir}")
        return
    
    logger.info(f"처리할 이미지 파일 {len(image_files)}개 발견")
    
    import cv2
    processed_count = 0
    
    for image_file in image_files:
        try:
            logger.info(f"처리 중: {image_file.name}")
            
            # 이미지 로드
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"이미지 로드 실패: {image_file}")
                continue
            
            # 얼굴 검출
            detection_result = detection_service.detect_faces(image)
            
            # 결과 저장
            result_data = {
                'image_file': image_file.name,
                'faces_count': len(detection_result.faces),
                'processing_time_ms': detection_result.processing_time_ms,
                'faces': []
            }
            
            # 각 얼굴에 대해 인식 수행
            for i, face in enumerate(detection_result.faces):
                # 얼굴 영역 추출
                x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
                face_image = image[y:y+h, x:x+w]
                
                # 임베딩 추출
                embedding = recognition_service.extract_embedding(face_image)
                face.embedding = embedding
                
                # 인물 식별
                identified_person = recognition_service.identify_face(face)
                
                face_data = {
                    'face_id': face.face_id,
                    'bbox': [x, y, w, h],
                    'confidence': float(face.confidence.value),
                    'quality_score': face.quality_score,
                    'identified_person': {
                        'person_id': identified_person.person_id if identified_person else None,
                        'name': identified_person.name if identified_person else 'Unknown'
                    }
                }
                result_data['faces'].append(face_data)
            
            # 결과 파일 저장
            result_file = output_path / f"{image_file.stem}_result.json"
            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            logger.info(f"처리 완료: {image_file.name} ({len(detection_result.faces)}개 얼굴)")
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 ({image_file.name}): {str(e)}")
            continue
    
    logger.info(f"배치 처리 완료 - 총 {processed_count}/{len(image_files)}개 파일 처리")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="얼굴 인식 시스템 실행기")
    parser.add_argument("--config", type=str, default="config/face_recognition_config.yaml",
                       help="설정 파일 경로")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch", "api"],
                       default="interactive", help="실행 모드")
    parser.add_argument("--input-dir", type=str, help="배치 모드 입력 디렉토리")
    parser.add_argument("--output-dir", type=str, help="배치 모드 출력 디렉토리")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API 모드 호스트")
    parser.add_argument("--port", type=int, default=8000, help="API 모드 포트")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)
    
    logger.info("🚀 얼굴 인식 시스템 시작")
    logger.info(f"실행 모드: {args.mode}")
    
    try:
        # 설정 로드
        config = load_config(args.config)
        logger.info(f"설정 파일 로드: {args.config}")
        
        # 서비스 초기화
        detection_service = create_face_detection_service(config)
        recognition_service = create_face_recognition_service(config)
        
        # 모드별 실행
        if args.mode == "interactive":
            run_interactive_demo(detection_service, recognition_service)
            
        elif args.mode == "batch":
            if not args.input_dir or not args.output_dir:
                logger.error("배치 모드에는 --input-dir과 --output-dir이 필요합니다")
                return 1
            
            run_batch_processing(detection_service, recognition_service,
                               args.input_dir, args.output_dir)
            
        elif args.mode == "api":
            # API 서버 실행
            from domains.face_recognition.interfaces.api.face_recognition_api import FaceRecognitionAPI
            
            api = FaceRecognitionAPI(config_path=args.config)
            logger.info(f"API 서버 시작: http://{args.host}:{args.port}")
            api.run(host=args.host, port=args.port, debug=args.debug)
        
        logger.info("✅ 시스템 종료")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중단됨")
        return 0
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 