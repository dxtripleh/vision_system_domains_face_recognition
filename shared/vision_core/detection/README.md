# Detection 모듈

이 모듈은 비전 시스템의 객체 검출 기능을 제공하는 공통 모듈입니다.

## 📁 폴더 구조

```
detection/
├── __init__.py              # 모듈 초기화
├── README.md                # 이 파일
├── base_detector.py         # 기본 검출기 클래스
├── face_detector.py         # 얼굴 검출기
├── object_detector.py       # 일반 객체 검출기
└── quality_assessor.py      # 품질 평가기
```

## 🎯 주요 기능

### 1. BaseDetector (기본 검출기)
```python
from shared.vision_core.detection.base_detector import BaseDetector

class CustomDetector(BaseDetector):
    """사용자 정의 검출기"""
    
    def __init__(self, model_path: str, config: Dict):
        super().__init__(model_path, config)
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """이미지에서 객체 검출"""
        # 검출 로직 구현
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """전처리"""
        # 전처리 로직 구현
        pass
    
    def postprocess(self, predictions: np.ndarray) -> List[Detection]:
        """후처리"""
        # 후처리 로직 구현
        pass
```

### 2. FaceDetector (얼굴 검출기)
```python
from shared.vision_core.detection.face_detector import FaceDetector

# 얼굴 검출기 초기화
detector = FaceDetector(
    model_path="models/weights/face_detection.onnx",
    config={
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "min_face_size": 80
    }
)

# 얼굴 검출
faces = detector.detect(image)

# 검출 결과 처리
for face in faces:
    bbox = face.bbox  # [x, y, w, h]
    confidence = face.confidence
    landmarks = face.landmarks  # 5점 랜드마크
```

### 3. ObjectDetector (일반 객체 검출기)
```python
from shared.vision_core.detection.object_detector import ObjectDetector

# 객체 검출기 초기화
detector = ObjectDetector(
    model_path="models/weights/object_detection.onnx",
    config={
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "classes": ["person", "car", "bicycle"]
    }
)

# 객체 검출
objects = detector.detect(image)

# 검출 결과 처리
for obj in objects:
    bbox = obj.bbox
    class_id = obj.class_id
    class_name = obj.class_name
    confidence = obj.confidence
```

### 4. QualityAssessor (품질 평가기)
```python
from shared.vision_core.detection.quality_assessor import QualityAssessor

# 품질 평가기 초기화
assessor = QualityAssessor(
    model_path="models/weights/quality_assessment.onnx",
    config={
        "blur_threshold": 0.3,
        "brightness_range": (0.2, 0.8),
        "contrast_range": (0.3, 0.7)
    }
)

# 이미지 품질 평가
quality_score = assessor.assess(image)

# 품질 메트릭
print(f"Blur Score: {quality_score.blur_score}")
print(f"Brightness Score: {quality_score.brightness_score}")
print(f"Contrast Score: {quality_score.contrast_score}")
print(f"Overall Score: {quality_score.overall_score}")
```

## 🔧 사용 예시

### 기본 검출 파이프라인
```python
import cv2
import numpy as np
from shared.vision_core.detection import FaceDetector, ObjectDetector

def detection_pipeline(image_path: str):
    """검출 파이프라인"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 얼굴 검출기 초기화
    face_detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    
    # 객체 검출기 초기화
    object_detector = ObjectDetector(
        model_path="models/weights/object_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    
    # 검출 실행
    faces = face_detector.detect(image)
    objects = object_detector.detect(image)
    
    # 결과 시각화
    result_image = image.copy()
    
    # 얼굴 바운딩 박스 그리기
    for face in faces:
        x, y, w, h = face.bbox
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, f"Face: {face.confidence:.2f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 객체 바운딩 박스 그리기
    for obj in objects:
        x, y, w, h = obj.bbox
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(result_image, f"{obj.class_name}: {obj.confidence:.2f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return result_image, faces, objects

# 사용 예시
if __name__ == "__main__":
    result_image, faces, objects = detection_pipeline("test_image.jpg")
    
    print(f"검출된 얼굴 수: {len(faces)}")
    print(f"검출된 객체 수: {len(objects)}")
    
    cv2.imshow("Detection Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 실시간 검출 시스템
```python
import cv2
import time
from shared.vision_core.detection import FaceDetector

def real_time_detection(camera_id: int = 0):
    """실시간 얼굴 검출"""
    
    # 카메라 초기화
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {camera_id}")
    
    # 얼굴 검출기 초기화
    detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
            
            # 얼굴 검출
            start_time = time.time()
            faces = detector.detect(frame)
            detection_time = time.time() - start_time
            
            # 결과 시각화
            for face in faces:
                x, y, w, h = face.bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {face.confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # FPS 표시
            fps = 1.0 / detection_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 화면 표시
            cv2.imshow("Real-time Face Detection", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    real_time_detection()
```

## 📊 성능 최적화

### 1. 배치 처리
```python
def batch_detection(detector, images: List[np.ndarray]) -> List[List[Detection]]:
    """배치 검출"""
    return detector.detect_batch(images)
```

### 2. GPU 가속
```python
# GPU 사용 설정
detector = FaceDetector(
    model_path="models/weights/face_detection.onnx",
    config={
        "device": "cuda",  # GPU 사용
        "precision": "fp16"  # 반정밀도 사용
    }
)
```

### 3. 멀티스레딩
```python
import threading
from concurrent.futures import ThreadPoolExecutor

def parallel_detection(detector, images: List[np.ndarray]) -> List[List[Detection]]:
    """병렬 검출"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(detector.detect, images))
    return results
```

## 🔧 설정 옵션

### 공통 설정
```python
COMMON_CONFIG = {
    "confidence_threshold": 0.5,    # 신뢰도 임계값
    "nms_threshold": 0.4,          # NMS 임계값
    "device": "cpu",               # 실행 디바이스 (cpu/cuda)
    "precision": "fp32",           # 정밀도 (fp32/fp16)
    "batch_size": 1,               # 배치 크기
    "num_threads": 4               # 스레드 수
}
```

### 얼굴 검출 설정
```python
FACE_DETECTION_CONFIG = {
    **COMMON_CONFIG,
    "min_face_size": 80,           # 최소 얼굴 크기
    "max_face_size": 640,          # 최대 얼굴 크기
    "landmark_points": 5,          # 랜드마크 점 수
    "enable_landmarks": True,      # 랜드마크 검출 활성화
    "enable_pose": False           # 자세 추정 비활성화
}
```

### 객체 검출 설정
```python
OBJECT_DETECTION_CONFIG = {
    **COMMON_CONFIG,
    "classes": ["person", "car", "bicycle"],  # 검출 클래스
    "input_size": (640, 640),      # 입력 크기
    "enable_tracking": False,      # 추적 비활성화
    "max_detections": 100          # 최대 검출 수
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
- 실시간 처리 시 프레임 스킵 고려

### 3. 정확도 vs 속도 트레이드오프
- 신뢰도 임계값 조정으로 정확도/속도 균형
- 입력 이미지 크기 조정으로 성능 최적화
- 모델 복잡도에 따른 하드웨어 요구사항 확인

## 📞 지원

### 문제 해결
1. **검출 성능 저하**: 모델 경로 및 설정 확인
2. **메모리 부족**: 배치 크기 및 이미지 크기 조정
3. **GPU 오류**: CUDA 버전 및 드라이버 확인
4. **검출 누락**: 신뢰도 임계값 조정

### 추가 도움말
- 각 검출기의 `__init__` 메서드 문서 참조
- 모델 파일의 메타데이터 확인
- 성능 벤치마크 결과 참조 