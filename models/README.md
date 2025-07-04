# Models 폴더

이 폴더는 비전 시스템에서 사용하는 AI 모델들을 저장하고 관리합니다.

## 📁 폴더 구조

```
models/
├── __init__.py                 # Python 패키지 초기화
├── README.md                   # 이 파일
├── weights/                    # 모델 가중치 파일 (.onnx)
│   ├── __init__.py
│   ├── face_detection_retinaface_widerface_20250628.onnx
│   ├── face_recognition_arcface_glint360k_20250628.onnx
│   └── defect_detection_yolov8n_factory_20250628.onnx
├── metadata/                   # 모델 메타데이터
│   ├── __init__.py
│   ├── face_detection_metadata.json
│   ├── face_recognition_metadata.json
│   └── defect_detection_metadata.json
└── configs/                    # 모델 설정 파일
    ├── __init__.py
    ├── face_detection_config.yaml
    ├── face_recognition_config.yaml
    └── defect_detection_config.yaml
```

## 🎯 주요 기능

### 1. 모델 가중치 저장
- **ONNX 형식**: 모든 모델은 ONNX 형식으로 저장 (크로스 플랫폼 호환성)
- **버전 관리**: 날짜 기반 버전 관리 시스템
- **압축 저장**: 대용량 모델 파일 압축 저장

### 2. 모델 메타데이터 관리
- **성능 정보**: 정확도, 속도, 메모리 사용량 등
- **학습 정보**: 데이터셋, 하이퍼파라미터, 학습 시간 등
- **호환성 정보**: 지원 플랫폼, 하드웨어 요구사항 등

### 3. 모델 설정 관리
- **추론 설정**: 입력 크기, 배치 크기, 정밀도 등
- **후처리 설정**: NMS, 필터링, 임계값 등
- **최적화 설정**: TensorRT, OpenVINO 등

## 📝 모델 네이밍 규칙

### 패턴: `{task}_{architecture}_{dataset}_{date}.onnx`

**지원 태스크**:
- `face_detection` (얼굴 감지)
- `face_recognition` (얼굴 인식)
- `emotion` (감정 인식)
- `landmark` (랜드마크 추출)
- `pose` (자세 추정)
- `tracking` (객체 추적)
- `defect_detection` (불량 검출)
- `powerline_inspection` (활선 검사)

**지원 아키텍처**:
- `retinaface`, `mtcnn`, `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- `arcface`, `facenet`, `sphereface`, `cosface`
- `efficientdet`, `ssd`, `faster_rcnn`

**예시**:
```
face_detection_retinaface_widerface_20250628.onnx
face_recognition_arcface_glint360k_20250628.onnx
defect_detection_yolov8n_factory_20250628.onnx
powerline_inspection_efficientdet_d1_20250628.onnx
```

## 🔧 사용 예시

### 모델 로딩
```python
import onnxruntime as ort
from pathlib import Path

def load_model(model_name: str, device: str = "auto"):
    """모델 로딩"""
    
    # 모델 경로
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "weights" / f"{model_name}.onnx"
    
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 실행 제공자 설정
    providers = []
    if device == "auto":
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    
    # 모델 세션 생성
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    return session

# 사용 예시
face_detection_model = load_model("face_detection_retinaface_widerface_20250628")
face_recognition_model = load_model("face_recognition_arcface_glint360k_20250628")
```

### 모델 메타데이터 확인
```python
import json
from pathlib import Path

def get_model_metadata(model_name: str):
    """모델 메타데이터 조회"""
    
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "models" / "metadata" / f"{model_name}_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata

# 사용 예시
face_detection_meta = get_model_metadata("face_detection_retinaface_widerface_20250628")
if face_detection_meta:
    print(f"모델 버전: {face_detection_meta['version']}")
    print(f"정확도: {face_detection_meta['accuracy']}")
    print(f"추론 속도: {face_detection_meta['inference_time']}ms")
```

### 모델 설정 로딩
```python
import yaml
from pathlib import Path

def load_model_config(model_name: str):
    """모델 설정 로딩"""
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / "models" / "configs" / f"{model_name}_config.yaml"
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# 사용 예시
face_detection_config = load_model_config("face_detection_retinaface_widerface_20250628")
if face_detection_config:
    print(f"입력 크기: {face_detection_config['input_size']}")
    print(f"신뢰도 임계값: {face_detection_config['confidence_threshold']}")
```

## 📊 모델 성능 모니터링

### 성능 메트릭 추적
```python
import time
import psutil
import numpy as np

def benchmark_model(model_session, test_images, num_runs=100):
    """모델 성능 벤치마크"""
    
    # 워밍업
    for _ in range(10):
        _ = model_session.run(None, {"input": test_images[0]})
    
    # 성능 측정
    inference_times = []
    memory_usage = []
    
    for _ in range(num_runs):
        # 메모리 사용량 측정
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 추론 시간 측정
        start_time = time.time()
        _ = model_session.run(None, {"input": test_images[0]})
        inference_time = (time.time() - start_time) * 1000  # ms
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        inference_times.append(inference_time)
        memory_usage.append(memory_after - memory_before)
    
    # 통계 계산
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    p95_inference_time = np.percentile(inference_times, 95)
    avg_memory_usage = np.mean(memory_usage)
    
    return {
        "avg_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "p95_inference_time_ms": p95_inference_time,
        "avg_memory_usage_mb": avg_memory_usage,
        "fps": 1000 / avg_inference_time
    }

# 사용 예시
test_image = np.random.rand(1, 3, 640, 640).astype(np.float32)
performance = benchmark_model(face_detection_model, test_image)
print(f"평균 추론 시간: {performance['avg_inference_time_ms']:.2f}ms")
print(f"FPS: {performance['fps']:.1f}")
print(f"평균 메모리 사용량: {performance['avg_memory_usage_mb']:.1f}MB")
```

## 🔄 모델 버전 관리

### 버전 비교
```python
def compare_model_versions(model_name: str, version1: str, version2: str):
    """모델 버전 비교"""
    
    meta1 = get_model_metadata(f"{model_name}_{version1}")
    meta2 = get_model_metadata(f"{model_name}_{version2}")
    
    if not meta1 or not meta2:
        return None
    
    comparison = {
        "accuracy_diff": meta2["accuracy"] - meta1["accuracy"],
        "speed_diff": meta1["inference_time"] - meta2["inference_time"],
        "size_diff": meta2["model_size_mb"] - meta1["model_size_mb"],
        "improvements": [],
        "regressions": []
    }
    
    # 개선사항 및 퇴보사항 분석
    if comparison["accuracy_diff"] > 0:
        comparison["improvements"].append(f"정확도 {comparison['accuracy_diff']:.3f} 향상")
    elif comparison["accuracy_diff"] < 0:
        comparison["regressions"].append(f"정확도 {abs(comparison['accuracy_diff']):.3f} 감소")
    
    if comparison["speed_diff"] > 0:
        comparison["improvements"].append(f"속도 {comparison['speed_diff']:.1f}ms 향상")
    elif comparison["speed_diff"] < 0:
        comparison["regressions"].append(f"속도 {abs(comparison['speed_diff']):.1f}ms 감소")
    
    return comparison

# 사용 예시
comparison = compare_model_versions(
    "face_detection_retinaface_widerface",
    "20250628",
    "20250629"
)
if comparison:
    print("개선사항:", comparison["improvements"])
    print("퇴보사항:", comparison["regressions"])
```

## 🚨 주의사항

### 1. 모델 호환성
- **ONNX 형식 필수**: PyTorch 모델은 ONNX로 변환 후 저장
- **버전 호환성**: ONNX Runtime 버전과 모델 버전 호환성 확인
- **플랫폼 호환성**: 크로스 플랫폼 호환성 검증

### 2. 성능 고려사항
- **메모리 사용량**: GPU 메모리 부족 시 모델 크기 조정
- **추론 속도**: 실시간 처리 시 FPS 요구사항 확인
- **정확도 vs 속도**: 트레이드오프 고려

### 3. 보안 고려사항
- **모델 보호**: 민감한 모델 파일 암호화 고려
- **접근 제어**: 모델 파일 접근 권한 관리
- **백업**: 중요 모델 파일 정기 백업

### 4. 유지보수 고려사항
- **버전 관리**: 명확한 버전 관리 시스템 구축
- **문서화**: 모델 성능 및 사용법 문서화
- **테스트**: 정기적인 모델 성능 테스트

## 📞 지원

### 문제 해결
1. **모델 로딩 실패**: 파일 경로 및 형식 확인
2. **성능 저하**: 하드웨어 및 설정 최적화
3. **메모리 부족**: 모델 크기 및 배치 크기 조정
4. **호환성 오류**: ONNX Runtime 버전 확인

### 추가 도움말
- 각 모델의 메타데이터 파일 참조
- 성능 벤치마크 결과 확인
- 모델 변환 가이드 참조
- GitHub Issues에서 유사한 문제 검색

### 기여 방법
1. 새로운 모델 추가 시 네이밍 규칙 준수
2. 모델 메타데이터 및 설정 파일 포함
3. 성능 벤치마크 결과 제공
4. 문서화 및 사용 예시 포함 