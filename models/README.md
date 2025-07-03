# 📁 MODELS 폴더 - AI 모델 저장소

## 🎯 **목적**
AI 모델의 가중치, 메타데이터, 설정 파일을 체계적으로 관리합니다.
얼굴인식을 시작으로 향후 확장될 다양한 비전 AI 모델들을 저장합니다.

## 📂 **구조**
```
models/
├── weights/             # 훈련된 모델 가중치 파일들
│   └── face_detection_opencv_haarcascade_20250628.xml
├── metadata/            # 모델 메타데이터 (성능, 버전 정보)
│   └── opencv_haarcascade_metadata.yaml
├── configs/             # 모델별 설정 파일들
└── new_models/          # 새로 추가될 모델들 (임시)
```

## 🚀 **주요 기능**

### 1. **모델 가중치 관리 (weights/)**
훈련된 AI 모델의 가중치 파일들을 저장

**지원 형식**:
- `.pt` - PyTorch 모델
- `.onnx` - ONNX 호환 모델 (추론 최적화)
- `.h5` - Keras/TensorFlow 모델
- `.xml/.bin` - OpenVINO 모델
- `.xml` - OpenCV Cascade 모델

**네이밍 규칙**:
```
{task}_{architecture}_{dataset}_{date}.{ext}

예시:
- face_detection_retinaface_widerface_20250628.pt
- face_recognition_arcface_glint360k_20250628.onnx
- emotion_resnet50_fer_20250628.h5
```

### 2. **메타데이터 관리 (metadata/)**
각 모델의 성능, 버전, 사용법 정보를 저장

```yaml
# opencv_haarcascade_metadata.yaml 예시
model_info:
  name: "OpenCV Haar Cascade Face Detector"
  version: "4.8.0"
  task: "face_detection"
  architecture: "haar_cascade"
  
performance:
  accuracy: 0.85
  fps: 30
  memory_usage: "50MB"
  
requirements:
  opencv_version: ">=4.5.0"
  python_version: ">=3.8"
  
usage:
  input_size: [640, 480]
  confidence_threshold: 0.3
  nms_threshold: 0.4
```

### 3. **설정 파일 관리 (configs/)**
모델별 세부 설정 및 하이퍼파라미터를 저장

## 🔄 **모델 생명주기**

### 1. **모델 추가 과정**
```bash
# 1. 새 모델을 new_models/에 임시 저장
cp new_model.pt models/new_models/

# 2. 검증 및 테스트
python tools/validation/validate_model.py models/new_models/new_model.pt

# 3. 메타데이터 생성
python tools/model_management/generate_metadata.py models/new_models/new_model.pt

# 4. 정식 등록 (weights/로 이동)
python tools/model_management/register_model.py models/new_models/new_model.pt
```

### 2. **모델 버전 관리**
```python
VERSION_NAMING = {
    'major': '새로운 아키텍처나 대규모 변경',
    'minor': '성능 개선이나 기능 추가', 
    'patch': '버그 수정이나 소규모 개선'
}

# 예시: face_detection_yolov8_v2.1.3_20250628.pt
```

### 3. **모델 선택 전략**
```python
MODEL_SELECTION_CRITERIA = {
    'accuracy_first': '정확도 우선 (서버 환경)',
    'speed_first': '속도 우선 (실시간 처리)',
    'balanced': '균형 (일반적 사용)',
    'mobile_optimized': '모바일 최적화'
}
```

## 📝 **사용 가이드라인**

### ✅ **허용되는 것들**
- 훈련된 AI 모델 가중치
- 모델 메타데이터 및 성능 정보
- 모델별 설정 파일
- 추론 최적화된 모델 (ONNX, OpenVINO)

### ❌ **금지되는 것들**
- 훈련 데이터 (→ `datasets/`로 이동)
- 소스 코드 (→ 해당 도메인으로 이동)
- 임시 결과물 (→ `data/temp/`로 이동)
- 사용자 데이터 (→ `data/`로 이동)

## 🔗 **관련 문서**
- [프로젝트 개요](../README.md)
- [구조 문서](STRUCTURE.md)
- [모델 관리 가이드](../docs/guides/MODEL_MANAGEMENT_GUIDE.md)
- [모델 다운로드 가이드](../tools/setup/README.md)

## 💡 **초보자 팁**

### 1. **모델 다운로드**
```bash
# 기본 모델들 자동 다운로드
python tools/setup/download_models.py

# 특정 모델만 다운로드
python tools/setup/download_models.py --model face_detection
```

### 2. **모델 로딩 예시**
```python
from domains.face_recognition.infrastructure.models.retinaface_detector import RetinaFaceDetector

# 모델 로딩
detector = RetinaFaceDetector(
    model_path="models/weights/face_detection_retinaface_widerface_20250628.pt",
    confidence_threshold=0.5
)

# 추론 실행
detections = detector.detect(image)
```

### 3. **모델 성능 확인**
```bash
# 모델 메타데이터 확인
python tools/model_management/show_model_info.py models/weights/face_detection_*.pt

# 모델 벤치마크
python tools/testing/benchmark_model.py models/weights/face_detection_*.pt
```

### 4. **모델 형식 변환**
```bash
# PyTorch → ONNX 변환
python tools/model_management/convert_to_onnx.py models/weights/model.pt

# ONNX → OpenVINO 변환  
python tools/model_management/convert_to_openvino.py models/weights/model.onnx
```

## ⚠️ **주의사항**

### 🚨 **절대 금지**
1. **모델 파일을 다른 폴더에 저장**
   ```
   ❌ domains/face_recognition/model.pt
   ✅ models/weights/face_detection_*.pt
   ```

2. **메타데이터 없는 모델 사용**
   ```
   ❌ 성능 정보 없이 모델 사용
   ✅ metadata/ 폴더의 정보 확인 후 사용
   ```

### 💾 **저장 공간 관리**
- **대용량 모델**: 클라우드 스토리지 활용 고려
- **버전 관리**: 오래된 버전은 정기적으로 정리
- **압축**: 사용하지 않는 모델은 압축 보관

### 🔒 **보안 고려사항**
- **라이센스 확인**: 상용 모델의 라이센스 준수
- **접근 권한**: 중요한 모델의 접근 권한 관리
- **백업**: 핵심 모델의 정기적 백업

### 📊 **성능 모니터링**
- **정기 검증**: 모델 성능 정기적 확인
- **버전 비교**: 새 버전과 기존 버전 성능 비교
- **하드웨어 최적화**: 배포 환경에 맞는 모델 선택

---
*이 문서는 프로젝트 구조 검증 시스템에 의해 자동 생성되고 수동으로 개선되었습니다.*
