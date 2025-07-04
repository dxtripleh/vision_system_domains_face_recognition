# Preprocessing 모듈

이 모듈은 비전 시스템의 이미지 전처리 기능을 제공하는 공통 모듈입니다.

## 📁 폴더 구조

```
preprocessing/
├── __init__.py              # 모듈 초기화
├── README.md                # 이 파일
├── image_processor.py       # 기본 이미지 처리기
├── face_aligner.py          # 얼굴 정렬기
├── augmentation.py          # 데이터 증강
└── normalization.py         # 정규화 처리
```

## 🎯 주요 기능

### 1. ImageProcessor (기본 이미지 처리기)
```python
from shared.vision_core.preprocessing.image_processor import ImageProcessor

# 이미지 처리기 초기화
processor = ImageProcessor(
    config={
        "resize": (640, 640),
        "normalize": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
)

# 이미지 전처리
image = cv2.imread("input.jpg")
processed_image = processor.process(image)

# 전처리 정보
print(f"원본 크기: {image.shape}")
print(f"처리 후 크기: {processed_image.shape}")
```

### 2. FaceAligner (얼굴 정렬기)
```python
from shared.vision_core.preprocessing.face_aligner import FaceAligner

# 얼굴 정렬기 초기화
aligner = FaceAligner(
    config={
        "output_size": (112, 112),
        "eye_center_ratio": 0.35,
        "enable_landmarks": True
    }
)

# 얼굴 이미지와 랜드마크로 정렬
face_image = cv2.imread("face.jpg")
landmarks = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]  # 5점 랜드마크

# 얼굴 정렬
aligned_face = aligner.align(face_image, landmarks)

# 정렬된 얼굴 저장
cv2.imwrite("aligned_face.jpg", aligned_face)
```

### 3. Augmentation (데이터 증강)
```python
from shared.vision_core.preprocessing.augmentation import Augmentation

# 데이터 증강기 초기화
augmenter = Augmentation(
    config={
        "rotation_range": (-15, 15),
        "brightness_range": (0.8, 1.2),
        "contrast_range": (0.8, 1.2),
        "flip_probability": 0.5,
        "noise_probability": 0.3
    }
)

# 단일 이미지 증강
image = cv2.imread("input.jpg")
augmented_images = augmenter.augment_single(image, num_augmentations=5)

# 배치 이미지 증강
image_batch = [cv2.imread(f"image_{i}.jpg") for i in range(10)]
augmented_batch = augmenter.augment_batch(image_batch, num_augmentations=3)
```

### 4. Normalization (정규화 처리)
```python
from shared.vision_core.preprocessing.normalization import Normalization

# 정규화 처리기 초기화
normalizer = Normalization(
    config={
        "method": "imagenet",  # imagenet, custom, minmax
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_range": (0, 255),
        "output_range": (0, 1)
    }
)

# 이미지 정규화
image = cv2.imread("input.jpg")
normalized_image = normalizer.normalize(image)

# 정규화 정보
print(f"정규화 전 범위: {image.min()} ~ {image.max()}")
print(f"정규화 후 범위: {normalized_image.min():.3f} ~ {normalized_image.max():.3f}")
```

## 🔧 사용 예시

### 기본 전처리 파이프라인
```python
import cv2
import numpy as np
from shared.vision_core.preprocessing import ImageProcessor, FaceAligner, Normalization

def preprocessing_pipeline(image_path: str, landmarks: List[List[int]] = None):
    """전처리 파이프라인"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 1. 기본 이미지 처리
    image_processor = ImageProcessor(
        config={
            "resize": (640, 640),
            "normalize": False,  # 정규화는 별도로 수행
            "interpolation": cv2.INTER_LINEAR
        }
    )
    processed_image = image_processor.process(image)
    
    # 2. 얼굴 정렬 (랜드마크가 있는 경우)
    if landmarks is not None:
        face_aligner = FaceAligner(
            config={
                "output_size": (112, 112),
                "eye_center_ratio": 0.35
            }
        )
        processed_image = face_aligner.align(processed_image, landmarks)
    
    # 3. 정규화
    normalizer = Normalization(
        config={
            "method": "imagenet",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    )
    normalized_image = normalizer.normalize(processed_image)
    
    return normalized_image

# 사용 예시
if __name__ == "__main__":
    # 랜드마크가 있는 경우
    landmarks = [[100, 120], [150, 120], [125, 140], [110, 160], [140, 160]]
    processed_image = preprocessing_pipeline("face.jpg", landmarks)
    
    # 랜드마크가 없는 경우
    processed_image = preprocessing_pipeline("general_image.jpg")
    
    print(f"전처리 완료: {processed_image.shape}")
```

### 데이터 증강 파이프라인
```python
import cv2
import os
from pathlib import Path
from shared.vision_core.preprocessing import Augmentation

def augmentation_pipeline(input_dir: str, output_dir: str, num_augmentations: int = 5):
    """데이터 증강 파이프라인"""
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 증강기 초기화
    augmenter = Augmentation(
        config={
            "rotation_range": (-15, 15),
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.8, 1.2),
            "flip_probability": 0.5,
            "noise_probability": 0.3,
            "blur_probability": 0.2,
            "sharpness_range": (0.8, 1.2)
        }
    )
    
    # 입력 이미지 목록
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    total_augmented = 0
    
    for image_file in image_files:
        # 이미지 로드
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # 증강 수행
        augmented_images = augmenter.augment_single(image, num_augmentations)
        
        # 증강된 이미지 저장
        base_name = image_file.stem
        for i, aug_image in enumerate(augmented_images):
            output_file = output_path / f"{base_name}_aug_{i:03d}.jpg"
            cv2.imwrite(str(output_file), aug_image)
            total_augmented += 1
        
        print(f"처리 완료: {image_file.name} -> {num_augmentations}개 증강")
    
    print(f"총 증강된 이미지 수: {total_augmented}")

# 사용 예시
if __name__ == "__main__":
    augmentation_pipeline(
        input_dir="datasets/humanoid/raw",
        output_dir="datasets/humanoid/augmented",
        num_augmentations=5
    )
```

### 실시간 전처리 시스템
```python
import cv2
import time
from shared.vision_core.preprocessing import ImageProcessor, Normalization

def real_time_preprocessing(camera_id: int = 0):
    """실시간 전처리"""
    
    # 카메라 초기화
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {camera_id}")
    
    # 전처리기 초기화
    image_processor = ImageProcessor(
        config={
            "resize": (224, 224),
            "normalize": False
        }
    )
    
    normalizer = Normalization(
        config={
            "method": "imagenet",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    )
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
            
            # 전처리 수행
            start_time = time.time()
            
            # 1. 리사이즈
            resized_frame = image_processor.process(frame)
            
            # 2. 정규화
            normalized_frame = normalizer.normalize(resized_frame)
            
            processing_time = time.time() - start_time
            
            # 원본과 전처리 결과 표시
            # 원본을 원래 크기로 리사이즈
            display_original = cv2.resize(frame, (400, 300))
            display_processed = cv2.resize(resized_frame, (400, 300))
            
            # 화면에 표시
            combined = np.hstack([display_original, display_processed])
            cv2.putText(combined, f"Processing Time: {processing_time*1000:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Real-time Preprocessing", combined)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# 사용 예시
if __name__ == "__main__":
    real_time_preprocessing()
```

### 얼굴 정렬 파이프라인
```python
import cv2
import numpy as np
from pathlib import Path
from shared.vision_core.detection import FaceDetector
from shared.vision_core.preprocessing import FaceAligner

def face_alignment_pipeline(input_dir: str, output_dir: str):
    """얼굴 정렬 파이프라인"""
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 얼굴 검출기 초기화
    detector = FaceDetector(
        model_path="models/weights/face_detection.onnx",
        config={"confidence_threshold": 0.5}
    )
    
    # 얼굴 정렬기 초기화
    aligner = FaceAligner(
        config={
            "output_size": (112, 112),
            "eye_center_ratio": 0.35,
            "enable_landmarks": True
        }
    )
    
    # 입력 이미지 목록
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    aligned_count = 0
    failed_count = 0
    
    for image_file in image_files:
        # 이미지 로드
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # 얼굴 검출
        faces = detector.detect(image)
        
        if len(faces) == 0:
            print(f"얼굴 검출 실패: {image_file.name}")
            failed_count += 1
            continue
        
        # 첫 번째 얼굴만 처리 (여러 얼굴이 있는 경우)
        face = faces[0]
        
        if face.landmarks is None:
            print(f"랜드마크 없음: {image_file.name}")
            failed_count += 1
            continue
        
        try:
            # 얼굴 정렬
            aligned_face = aligner.align(image, face.landmarks)
            
            # 정렬된 얼굴 저장
            output_file = output_path / f"aligned_{image_file.name}"
            cv2.imwrite(str(output_file), aligned_face)
            
            aligned_count += 1
            print(f"정렬 완료: {image_file.name}")
            
        except Exception as e:
            print(f"정렬 실패: {image_file.name} - {str(e)}")
            failed_count += 1
    
    print(f"정렬 완료: {aligned_count}개")
    print(f"실패: {failed_count}개")

# 사용 예시
if __name__ == "__main__":
    face_alignment_pipeline(
        input_dir="datasets/humanoid/raw",
        output_dir="datasets/humanoid/aligned"
    )
```

## 📊 성능 최적화

### 1. 배치 처리
```python
def batch_preprocessing(processor, images: List[np.ndarray]) -> List[np.ndarray]:
    """배치 전처리"""
    return processor.process_batch(images)
```

### 2. GPU 가속
```python
# GPU 사용 설정
processor = ImageProcessor(
    config={
        "device": "cuda",  # GPU 사용
        "precision": "fp16"  # 반정밀도 사용
    }
)
```

### 3. 메모리 최적화
```python
def memory_efficient_preprocessing(image_path: str):
    """메모리 효율적인 전처리"""
    # 이미지를 청크 단위로 처리
    chunk_size = 1024
    
    with open(image_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            # 청크 처리
            process_chunk(chunk)
```

## 🔧 설정 옵션

### 이미지 처리 설정
```python
IMAGE_PROCESSING_CONFIG = {
    "resize": (640, 640),           # 리사이즈 크기
    "interpolation": cv2.INTER_LINEAR,  # 보간법
    "normalize": True,              # 정규화 여부
    "mean": [0.485, 0.456, 0.406],  # 평균값
    "std": [0.229, 0.224, 0.225],   # 표준편차
    "device": "cpu",                # 실행 디바이스
    "precision": "fp32",            # 정밀도
    "enable_cache": True,           # 캐시 활성화
    "cache_size": 100               # 캐시 크기
}
```

### 얼굴 정렬 설정
```python
FACE_ALIGNMENT_CONFIG = {
    "output_size": (112, 112),      # 출력 크기
    "eye_center_ratio": 0.35,       # 눈 중심 비율
    "enable_landmarks": True,       # 랜드마크 활성화
    "landmark_points": 5,           # 랜드마크 점 수
    "interpolation": cv2.INTER_LINEAR,  # 보간법
    "border_mode": cv2.BORDER_CONSTANT,  # 경계 모드
    "border_value": 0,              # 경계 값
    "enable_quality_check": True,   # 품질 검사 활성화
    "min_face_size": 80             # 최소 얼굴 크기
}
```

### 데이터 증강 설정
```python
AUGMENTATION_CONFIG = {
    "rotation_range": (-15, 15),    # 회전 범위 (도)
    "brightness_range": (0.8, 1.2), # 밝기 범위
    "contrast_range": (0.8, 1.2),   # 대비 범위
    "saturation_range": (0.8, 1.2), # 채도 범위
    "hue_range": (-10, 10),         # 색조 범위 (도)
    "flip_probability": 0.5,        # 좌우 반전 확률
    "noise_probability": 0.3,       # 노이즈 추가 확률
    "blur_probability": 0.2,        # 블러 확률
    "sharpness_range": (0.8, 1.2),  # 선명도 범위
    "enable_color_jitter": True,    # 색상 지터 활성화
    "enable_geometric": True        # 기하학적 변환 활성화
}
```

### 정규화 설정
```python
NORMALIZATION_CONFIG = {
    "method": "imagenet",           # 정규화 방법
    "mean": [0.485, 0.456, 0.406],  # 평균값
    "std": [0.229, 0.224, 0.225],   # 표준편차
    "input_range": (0, 255),        # 입력 범위
    "output_range": (0, 1),         # 출력 범위
    "enable_clip": True,            # 클리핑 활성화
    "clip_range": (0, 1),           # 클리핑 범위
    "enable_histogram": False,      # 히스토그램 평활화 비활성화
    "enable_adaptive": False        # 적응형 정규화 비활성화
}
```

## 🚨 주의사항

### 1. 메모리 관리
- 대용량 이미지 처리 시 메모리 사용량 모니터링
- 배치 크기 조정으로 메모리 부족 방지
- 임시 파일 자동 정리 설정

### 2. 성능 고려사항
- GPU 사용 시 메모리 전송 오버헤드 고려
- 실시간 처리 시 전처리 시간 최적화
- 캐시 사용으로 반복 작업 최적화

### 3. 품질 유지
- 전처리 후 이미지 품질 검증
- 원본 데이터 백업 유지
- 전처리 파라미터 기록 및 버전 관리

### 4. 호환성 확인
- 모델 입력 형식과 전처리 결과 일치 확인
- 다양한 이미지 형식 지원 확인
- 플랫폼별 호환성 검증

## 📞 지원

### 문제 해결
1. **메모리 부족**: 배치 크기 및 이미지 크기 조정
2. **처리 속도 저하**: GPU 사용 및 캐시 활성화
3. **품질 저하**: 전처리 파라미터 조정
4. **호환성 오류**: 입력/출력 형식 확인

### 추가 도움말
- 각 전처리기의 `__init__` 메서드 문서 참조
- 모델 요구사항과 전처리 설정 일치 확인
- 성능 벤치마크 결과 참조 