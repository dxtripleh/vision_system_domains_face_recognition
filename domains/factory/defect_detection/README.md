# Factory Defect Detection

공장 불량 검출을 위한 도메인입니다. YOLOv8 기반 ONNX 모델을 사용하여 실시간 불량 검출을 수행합니다.

## 🏗️ 구조

```
domains/factory/defect_detection/
├── __init__.py          # 도메인 초기화
├── model.py             # ONNX 모델 클래스
├── run.py               # 실시간 추론 실행
├── test_model.py        # 단위 테스트
└── README.md            # 이 파일
```

## 🎯 기능

### 지원하는 불량 유형
- **스크래치** (Scratch): 제품 표면의 긁힘
- **함몰** (Dent): 제품 표면의 눌림
- **균열** (Crack): 제품의 깨짐
- **변색** (Discoloration): 색상 변화
- **오염** (Contamination): 이물질

### 주요 기능
- **실시간 검출**: USB 카메라 또는 이미지 파일 입력
- **크로스 플랫폼**: Windows, Ubuntu, Jetson 지원
- **하드웨어 최적화**: GPU/CPU 자동 감지 및 최적화
- **시각화**: 바운딩 박스, 클래스명, 신뢰도 표시
- **로깅**: 검출 결과 및 성능 통계 로그 저장

## 🚀 사용법

### 1. 카메라 실시간 검출
```bash
# USB 카메라 0번 사용
python domains/factory/defect_detection/run.py --source 0

# 특정 모델 파일 사용
python domains/factory/defect_detection/run.py --source 0 --model models/weights/defect_detection_yolov8n_factory.onnx

# 신뢰도 임계값 조정
python domains/factory/defect_detection/run.py --source 0 --conf 0.7

# 결과 저장
python domains/factory/defect_detection/run.py --source 0 --save
```

### 2. 이미지 파일 검출
```bash
# 이미지 파일 검출
python domains/factory/defect_detection/run.py --source path/to/image.jpg

# 화면 표시 없이 실행
python domains/factory/defect_detection/run.py --source path/to/image.jpg --no-display
```

### 3. 단위 테스트
```bash
# 기본 테스트 실행
python domains/factory/defect_detection/test_model.py

# unittest 모드로 실행
python domains/factory/defect_detection/test_model.py --unittest
```

## 🔧 설정

### 모델 설정
```python
config = {
    'confidence_threshold': 0.5,  # 신뢰도 임계값
    'nms_threshold': 0.4,         # NMS 임계값
    'input_size': (640, 640),     # 입력 이미지 크기
    'max_detections': 100         # 최대 검출 개수
}
```

### 품질 관리 임계값
```python
QUALITY_THRESHOLDS = {
    'alert_threshold': 0.05,     # 5% 불량률에서 경고
    'critical_threshold': 0.1,   # 10% 불량률에서 심각 알림
    'sampling_rate': 1.0         # 100% 샘플링
}
```

## 📊 출력 형식

### 검출 결과
```python
{
    'bbox': [x1, y1, x2, y2],      # 바운딩 박스 좌표
    'class_id': 0,                 # 클래스 ID
    'confidence': 0.85,            # 신뢰도
    'class_name': '스크래치'       # 클래스명
}
```

### 성능 통계
- **FPS**: 초당 프레임 수
- **Total Frames**: 총 처리 프레임 수
- **Total Defects**: 총 검출된 불량 수
- **Avg Inference**: 평균 추론 시간

## 🎮 키보드 단축키

실시간 실행 중 사용 가능한 단축키:
- **q**: 종료
- **s**: 현재 프레임 저장

## 📁 파일 위치

### 입력
- **카메라**: USB 카메라 (기본값: 0번)
- **이미지**: 지원 형식 (jpg, png, bmp)

### 출력
- **로그**: `data/logs/defect_detection.log`
- **이미지**: `data/output/defect_result_*.jpg`
- **비디오**: `data/output/defect_detection_*.mp4`

## 🔍 하드웨어 최적화

### 자동 감지
- **Jetson**: TensorRT 최적화
- **GPU**: CUDA 가속
- **CPU**: 기본 최적화

### 성능 권장사항
- **실시간**: Jetson Nano 이상 또는 RTX 3060 이상
- **배치 처리**: CPU 8코어 이상
- **메모리**: 8GB 이상 권장

## ⚠️ 주의사항

1. **모델 파일**: ONNX 형식의 YOLOv8 모델이 필요합니다
2. **하드웨어**: GPU 사용 시 CUDA 드라이버 설치 필요
3. **카메라**: USB 카메라 연결 확인
4. **권한**: Jetson 환경에서 카메라 접근 권한 필요

## 🔗 관련 문서

- [프로젝트 개요](../../../README.md)
- [크로스 플랫폼 호환성 규칙](../../../.cursor/rules/CROSS_PLATFORM_COMPATIBILITY.mdc)
- [공통 모듈](../../../shared/README.md) 