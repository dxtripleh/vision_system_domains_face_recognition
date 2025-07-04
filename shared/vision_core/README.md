# Vision Core 모듈

이 모듈은 비전 시스템의 핵심 알고리즘을 제공하는 공통 모듈입니다.

## 📁 폴더 구조

```
vision_core/
├── __init__.py              # 모듈 초기화
├── README.md                # 이 파일
├── detection/               # 객체 검출 모듈
│   ├── __init__.py
│   ├── README.md
│   ├── base_detector.py
│   ├── face_detector.py
│   ├── object_detector.py
│   └── quality_assessor.py
├── recognition/             # 인식 모듈
│   ├── __init__.py
│   ├── README.md
│   ├── base_recognizer.py
│   ├── face_embedder.py
│   ├── face_matcher.py
│   └── similarity_matcher.py
├── preprocessing/           # 전처리 모듈
│   ├── __init__.py
│   ├── README.md
│   ├── image_processor.py
│   ├── face_aligner.py
│   ├── augmentation.py
│   └── normalization.py
├── postprocessing/          # 후처리 모듈 (향후)
│   ├── __init__.py
│   ├── README.md
│   ├── nms_utils.py
│   ├── filtering.py
│   └── visualization.py
├── tracking/                # 추적 모듈 (향후)
│   ├── __init__.py
│   ├── README.md
│   ├── tracker.py
│   ├── motion_model.py
│   └── association.py
└── pose_estimation/         # 자세 추정 모듈 (향후)
    ├── __init__.py
    ├── README.md
    ├── pose_estimator.py
    └── keypoint_utils.py
```

## 🎯 주요 기능

### 1. Detection (객체 검출)
- **얼굴 검출**: RetinaFace, MTCNN 등 다양한 얼굴 검출 모델 지원
- **일반 객체 검출**: YOLO, SSD 등 범용 객체 검출 모델 지원
- **품질 평가**: 이미지 품질 자동 평가 및 필터링

### 2. Recognition (인식)
- **얼굴 인식**: ArcFace, FaceNet 등 고성능 얼굴 인식 모델 지원
- **특징 매칭**: 코사인 유사도, 유클리드 거리 등 다양한 매칭 알고리즘
- **유사도 검색**: 대용량 데이터베이스에서 효율적인 유사도 검색

### 3. Preprocessing (전처리)
- **이미지 처리**: 리사이즈, 정규화, 색상 변환 등 기본 처리
- **얼굴 정렬**: 랜드마크 기반 얼굴 정렬 및 정규화
- **데이터 증강**: 회전, 밝기 조정, 노이즈 추가 등 다양한 증강 기법

### 4. Postprocessing (후처리) - 향후
- **NMS**: Non-Maximum Suppression으로 중복 검출 제거
- **필터링**: 신뢰도, 크기, 위치 기반 결과 필터링
- **시각화**: 검출 결과 시각화 및 어노테이션

### 5. Tracking (추적) - 향후
- **객체 추적**: Kalman Filter, SORT 등 추적 알고리즘
- **모션 모델**: 객체 움직임 예측 및 모델링
- **연관 분석**: 프레임 간 객체 연관성 분석

### 6. Pose Estimation (자세 추정) - 향후
- **키포인트 검출**: 인체, 얼굴 키포인트 검출
- **자세 추정**: 2D/3D 자세 추정 및 분석
- **제스처 인식**: 손, 얼굴 제스처 인식

## 🔧 사용 예시

### 기본 비전 파이프라인
```python
import cv2
import numpy as np
from shared.vision_core.detection import FaceDetector
from shared.vision_core.recognition import FaceEmbedder, FaceMatcher
from shared.vision_core.preprocessing import ImageProcessor

def vision_pipeline(image_path: str):
    """완전한 비전 처리 파이프라인"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 1. 전처리
    processor = ImageProcessor(config={"resize": (640, 640)})
    processed_image = processor.process(image)
    
    # 2. 얼굴 검출
    detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    faces = detector.detect(processed_image)
    
    # 3. 얼굴 인식 (검출된 얼굴이 있는 경우)
    if faces:
        embedder = FaceEmbedder(
            model_path="models/weights/face_recognition.onnx",
            config={"embedding_size": 512}
        )
        
        matcher = FaceMatcher(config={"similarity_threshold": 0.6})
        
        # 첫 번째 얼굴 인식
        face = faces[0]
        face_roi = processed_image[face.bbox[1]:face.bbox[1]+face.bbox[3], 
                                  face.bbox[0]:face.bbox[0]+face.bbox[2]]
        
        embedding = embedder.extract_embedding(face_roi)
        
        # 데이터베이스와 매칭 (예시)
        database = {"person1": np.random.rand(512)}  # 실제로는 저장된 임베딩
        best_match = "Unknown"
        best_score = 0.0
        
        for person_id, stored_embedding in database.items():
            similarity = matcher.match(embedding, stored_embedding)
            if similarity > best_score:
                best_score = similarity
                best_match = person_id
        
        return {
            "faces_detected": len(faces),
            "recognized_person": best_match,
            "confidence": best_score
        }
    
    return {"faces_detected": 0, "recognized_person": "None", "confidence": 0.0}

# 사용 예시
if __name__ == "__main__":
    result = vision_pipeline("test_image.jpg")
    print(f"검출된 얼굴: {result['faces_detected']}개")
    print(f"인식 결과: {result['recognized_person']}")
    print(f"신뢰도: {result['confidence']:.3f}")
```

### 실시간 비전 시스템
```python
import cv2
import time
from shared.vision_core.detection import FaceDetector, ObjectDetector
from shared.vision_core.recognition import FaceEmbedder, FaceMatcher
from shared.vision_core.preprocessing import ImageProcessor

def real_time_vision_system(camera_id: int = 0):
    """실시간 비전 시스템"""
    
    # 카메라 초기화
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {camera_id}")
    
    # 모듈 초기화
    image_processor = ImageProcessor(config={"resize": (640, 640)})
    face_detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    object_detector = ObjectDetector(
        model_path="models/weights/object_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
            
            # 전처리
            processed_frame = image_processor.process(frame)
            
            # 얼굴 검출
            faces = face_detector.detect(processed_frame)
            
            # 객체 검출
            objects = object_detector.detect(processed_frame)
            
            # 결과 시각화
            result_frame = processed_frame.copy()
            
            # 얼굴 바운딩 박스
            for face in faces:
                x, y, w, h = face.bbox
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_frame, f"Face: {face.confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 객체 바운딩 박스
            for obj in objects:
                x, y, w, h = obj.bbox
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_frame, f"{obj.class_name}: {obj.confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 통계 정보 표시
            cv2.putText(result_frame, f"Faces: {len(faces)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Objects: {len(objects)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 화면 표시
            cv2.imshow("Real-time Vision System", result_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    real_time_vision_system()
```

### 배치 처리 시스템
```python
import cv2
import numpy as np
from pathlib import Path
from shared.vision_core.detection import FaceDetector
from shared.vision_core.preprocessing import ImageProcessor, Augmentation

def batch_processing_pipeline(input_dir: str, output_dir: str):
    """배치 처리 파이프라인"""
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모듈 초기화
    processor = ImageProcessor(config={"resize": (640, 640)})
    detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    augmenter = Augmentation(
        config={
            "rotation_range": (-15, 15),
            "brightness_range": (0.8, 1.2),
            "flip_probability": 0.5
        }
    )
    
    # 입력 이미지 목록
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    total_processed = 0
    faces_detected = 0
    
    for image_file in image_files:
        # 이미지 로드
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # 전처리
        processed_image = processor.process(image)
        
        # 얼굴 검출
        faces = detector.detect(processed_image)
        
        if len(faces) > 0:
            faces_detected += len(faces)
            
            # 데이터 증강 (얼굴이 검출된 경우)
            augmented_images = augmenter.augment_single(processed_image, num_augmentations=3)
            
            # 결과 저장
            base_name = image_file.stem
            for i, aug_image in enumerate(augmented_images):
                output_file = output_path / f"{base_name}_aug_{i:03d}.jpg"
                cv2.imwrite(str(output_file), aug_image)
        
        total_processed += 1
        print(f"처리 완료: {image_file.name} (얼굴 {len(faces)}개)")
    
    print(f"총 처리된 이미지: {total_processed}개")
    print(f"총 검출된 얼굴: {faces_detected}개")

# 사용 예시
if __name__ == "__main__":
    batch_processing_pipeline(
        input_dir="datasets/humanoid/raw",
        output_dir="datasets/humanoid/processed"
    )
```

## 📊 성능 최적화

### 1. 모듈별 최적화
```python
# Detection 최적화
detector_config = {
    "device": "cuda",           # GPU 사용
    "precision": "fp16",        # 반정밀도
    "batch_size": 4,            # 배치 처리
    "num_threads": 4            # 멀티스레딩
}

# Recognition 최적화
recognizer_config = {
    "device": "cuda",
    "precision": "fp16",
    "enable_cache": True,       # 캐시 활성화
    "cache_size": 1000
}

# Preprocessing 최적화
processor_config = {
    "device": "cuda",
    "enable_cache": True,
    "batch_size": 8
}
```

### 2. 파이프라인 최적화
```python
def optimized_pipeline(images: List[np.ndarray]):
    """최적화된 파이프라인"""
    
    # 배치 전처리
    processor = ImageProcessor(config={"batch_size": 8})
    processed_batch = processor.process_batch(images)
    
    # 배치 검출
    detector = FaceDetector(config={"batch_size": 4})
    detection_batch = detector.detect_batch(processed_batch)
    
    # 배치 인식
    embedder = FaceEmbedder(config={"batch_size": 4})
    embedding_batch = embedder.extract_embeddings_batch(processed_batch)
    
    return detection_batch, embedding_batch
```

### 3. 메모리 최적화
```python
def memory_efficient_processing(image_paths: List[str]):
    """메모리 효율적인 처리"""
    
    chunk_size = 10  # 한 번에 처리할 이미지 수
    
    for i in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[i:i+chunk_size]
        
        # 청크 단위로 처리
        chunk_images = [cv2.imread(path) for path in chunk_paths]
        chunk_results = process_chunk(chunk_images)
        
        # 결과 저장 후 메모리 해제
        save_results(chunk_results)
        del chunk_images, chunk_results
```

## 🔧 설정 옵션

### 공통 설정
```python
COMMON_CONFIG = {
    "device": "cpu",               # 실행 디바이스 (cpu/cuda)
    "precision": "fp32",           # 정밀도 (fp32/fp16)
    "batch_size": 1,               # 배치 크기
    "num_threads": 4,              # 스레드 수
    "enable_cache": False,         # 캐시 활성화
    "cache_size": 100,             # 캐시 크기
    "enable_logging": True,        # 로깅 활성화
    "log_level": "INFO"            # 로그 레벨
}
```

### 모듈별 설정
```python
# Detection 모듈 설정
DETECTION_CONFIG = {
    **COMMON_CONFIG,
    "confidence_threshold": 0.5,    # 신뢰도 임계값
    "nms_threshold": 0.4,          # NMS 임계값
    "min_detection_size": 20,      # 최소 검출 크기
    "max_detections": 100          # 최대 검출 수
}

# Recognition 모듈 설정
RECOGNITION_CONFIG = {
    **COMMON_CONFIG,
    "embedding_size": 512,          # 임베딩 차원
    "similarity_threshold": 0.6,    # 유사도 임계값
    "distance_metric": "cosine",    # 거리 측정 방식
    "enable_face_quality": True     # 얼굴 품질 검사
}

# Preprocessing 모듈 설정
PREPROCESSING_CONFIG = {
    **COMMON_CONFIG,
    "resize": (640, 640),           # 리사이즈 크기
    "normalize": True,              # 정규화 여부
    "mean": [0.485, 0.456, 0.406],  # 평균값
    "std": [0.229, 0.224, 0.225]    # 표준편차
}
```

## 🚨 주의사항

### 1. 모델 호환성
- ONNX 모델만 지원 (PyTorch 모델은 변환 필요)
- 모델 입력/출력 형식 확인 필수
- 버전 호환성 검증 필요

### 2. 성능 고려사항
- GPU 메모리 부족 시 배치 크기 조정
- CPU 사용 시 스레드 수 최적화
- 실시간 처리 시 파이프라인 최적화

### 3. 메모리 관리
- 대용량 데이터 처리 시 메모리 사용량 모니터링
- 배치 크기 조정으로 메모리 부족 방지
- 임시 객체 자동 정리 설정

### 4. 정확도 vs 속도 트레이드오프
- 신뢰도 임계값 조정으로 정확도/속도 균형
- 모델 복잡도에 따른 하드웨어 요구사항 확인
- 실시간 처리 시 프레임 스킵 고려

## 📞 지원

### 문제 해결
1. **모델 로딩 실패**: 모델 경로 및 형식 확인
2. **메모리 부족**: 배치 크기 및 이미지 크기 조정
3. **GPU 오류**: CUDA 버전 및 드라이버 확인
4. **성능 저하**: 설정 최적화 및 하드웨어 확인

### 추가 도움말
- 각 모듈의 README.md 참조
- 모델 파일의 메타데이터 확인
- 성능 벤치마크 결과 참조
- GitHub Issues에서 유사한 문제 검색

### 기여 방법
1. 새로운 알고리즘 추가 시 해당 모듈에 구현
2. 기존 알고리즘 개선 시 문서화 추가
3. 성능 최적화 제안 시 벤치마크 결과 포함
4. 버그 리포트 시 상세한 환경 정보 포함 