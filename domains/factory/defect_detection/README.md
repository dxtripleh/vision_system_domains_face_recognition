# Defect Detection - 불량 검출 기능

## 📋 개요

이 모듈은 제조업 환경에서 제품의 불량을 자동으로 검출하고 분류하는 기능을 제공합니다. YOLO 기반 객체 검출 모델을 사용하여 실시간으로 불량을 감지하고, 품질 관리 시스템과 연동하여 생산 효율성을 향상시킵니다.

## 🎯 주요 기능

### 1. 실시간 불량 검출
- **다중 불량 유형 검출**: 스크래치, 얼룩, 균열, 찌그러짐 등
- **실시간 처리**: 25 FPS 이상의 고속 처리
- **높은 정확도**: 95% 이상의 검출 정확도
- **다중 카메라 지원**: 여러 카메라에서 동시 검출

### 2. 불량 분류 및 분석
- **자동 분류**: 검출된 불량의 유형 자동 분류
- **신뢰도 평가**: 검출 결과의 신뢰도 점수 제공
- **크기 측정**: 불량 영역의 크기 및 면적 계산
- **위치 추적**: 불량 발생 위치 기록

### 3. 품질 관리 시스템
- **실시간 모니터링**: 생산 라인 실시간 품질 모니터링
- **통계 분석**: 불량 발생 패턴 및 트렌드 분석
- **알림 시스템**: 불량률 임계값 초과 시 자동 알림
- **보고서 생성**: 일일/주간/월간 품질 보고서

## 🏗️ 모듈 구조

```
defect_detection/
├── __init__.py                    # 패키지 초기화
├── README.md                      # 이 파일
├── run_defect_detection.py        # 실시간 실행 스크립트
├── models/                        # 모델 클래스들
│   ├── __init__.py
│   └── defect_detection_model.py  # 불량 검출 모델
├── services/                      # 서비스 클래스들
│   └── __init__.py
├── utils/                         # 유틸리티 함수들
│   └── __init__.py
├── tests/                         # 테스트 파일들
│   ├── __init__.py
│   └── test_defect_detection_model.py  # 모델 테스트
├── configs/                       # 설정 파일들
│   └── __init__.py
└── pipeline/                      # 9단계 파이프라인
    └── __init__.py
```

## 🚀 빠른 시작

### 기본 실행
```bash
# 웹캠을 사용한 실시간 불량 검출
python run_defect_detection.py

# 이미지 파일로 테스트
python run_defect_detection.py --source path/to/product_image.jpg

# 결과 화면 표시
python run_defect_detection.py --show

# 결과 저장
python run_defect_detection.py --save

# 설정 파일 지정
python run_defect_detection.py --config config/defect_detection.yaml
```

### 명령줄 옵션
- `--source`: 입력 소스 (카메라 ID 또는 이미지/비디오 파일 경로)
- `--model`: 모델 파일 경로
- `--config`: 설정 파일 경로
- `--conf`: 신뢰도 임계값 (기본값: 0.5)
- `--show`: 결과 화면 표시
- `--save`: 결과 저장
- `--batch`: 배치 처리 모드

### 모델 테스트
```bash
# 모델 로딩 및 추론 테스트
python -m pytest tests/test_defect_detection_model.py -v

# 전체 테스트 실행
python -m pytest tests/ -v
```

## 🔧 기술 스택

### 핵심 기술
- **딥러닝 프레임워크**: PyTorch (학습), ONNX Runtime (추론)
- **객체 검출**: YOLOv8, YOLOv5, EfficientDet
- **컴퓨터 비전**: OpenCV
- **이미지 처리**: NumPy, PIL
- **데이터 관리**: JSON, YAML, SQLite

### 모델 아키텍처
- **기본 모델**: YOLOv8n (경량화)
- **고정확도 모델**: YOLOv8m (중간 크기)
- **최고정확도 모델**: YOLOv8l (대용량)

## 📊 불량 유형

### 지원하는 불량 유형
1. **Scratch (스크래치)**
   - 표면에 생긴 긁힘 자국
   - 신뢰도 임계값: 0.3
   - 색상 코드: [0, 0, 255] (빨강)

2. **Dent (찌그러짐)**
   - 표면이 눌려서 생긴 함몰
   - 신뢰도 임계값: 0.4
   - 색상 코드: [0, 255, 0] (초록)

3. **Crack (균열)**
   - 표면에 생긴 금이 가는 현상
   - 신뢰도 임계값: 0.5
   - 색상 코드: [255, 0, 0] (파랑)

4. **Discoloration (변색)**
   - 색상이 변하거나 얼룩진 현상
   - 신뢰도 임계값: 0.3
   - 색상 코드: [255, 255, 0] (노랑)

## 📈 성능 지표

### 검출 성능
- **검출 정확도**: 95% 이상
- **분류 정확도**: 90% 이상
- **처리 속도**: 25 FPS 이상 (GPU)
- **메모리 사용량**: 3GB 이하

### 하드웨어 요구사항
- **최소 사양**: CPU 4코어, RAM 8GB
- **권장 사양**: GPU 4GB+, RAM 16GB
- **최적 사양**: GPU 8GB+, RAM 32GB

## 🔧 설정 관리

### 기본 설정
```yaml
# config/defect_detection.yaml
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 100

classes:
  scratch: 0
  dent: 1
  crack: 2
  discoloration: 3

performance:
  target_fps: 25
  max_batch_size: 8

camera:
  device_id: 0
  resolution: [1920, 1080]
  fps: 30

output:
  save_results: true
  output_dir: "data/domains/factory/defect_detection/"
  save_images: true
  save_annotations: true

alerts:
  email_notification: false
  sound_alert: true
  defect_count_threshold: 10
```

### 환경별 설정
- **개발 환경**: `config/environments/development_defect_detection.yaml`
- **테스트 환경**: `config/environments/test_defect_detection.yaml`
- **운영 환경**: `config/environments/production_defect_detection.yaml`

## 🏭 공장 환경 고려사항

### 조명 조건
- **일정한 조명**: 검출 정확도를 위한 일정한 조명 환경 유지
- **그림자 최소화**: 제품 표면의 그림자 최소화
- **반사 방지**: 카메라 렌즈 반사 방지
- **조명 강도**: 500-1000 lux 권장

### 카메라 설치
- **적절한 각도**: 제품 표면에 수직 또는 45도 각도
- **적정 거리**: 제품 크기에 따른 적절한 촬영 거리
- **진동 방지**: 카메라 진동 방지 장치 설치
- **정기 보정**: 월 1회 카메라 보정 수행

### 네트워크 환경
- **안정적 연결**: 실시간 데이터 전송을 위한 안정적인 네트워크
- **대역폭**: 최소 100Mbps 권장
- **백업 시스템**: 네트워크 장애 시 백업 시스템 구축

## 📊 데이터 관리

### 학습 데이터
```
datasets/factory/defect_detection/
├── raw/                           # 원본 이미지
│   ├── scratch/                   # 스크래치 이미지
│   ├── dent/                      # 찌그러짐 이미지
│   ├── crack/                     # 균열 이미지
│   └── discoloration/             # 변색 이미지
├── processed/                     # 전처리된 이미지
├── annotations/                   # 라벨링 데이터 (YOLO 형식)
└── splits/                        # train/val/test 분할
```

### 런타임 데이터
```
data/domains/factory/defect_detection/
├── captures/                      # 캡처된 이미지
├── results/                       # 검출 결과
│   ├── images/                    # 결과 이미지
│   ├── annotations/               # 검출 주석
│   └── reports/                   # 검출 보고서
└── logs/                          # 실행 로그
```

## 🧪 테스트 전략

### 단위 테스트
- **모델 테스트**: 불량 검출 모델의 정확도 및 성능 검증
- **서비스 테스트**: 비즈니스 로직 검증
- **유틸리티 테스트**: 헬퍼 함수 검증

### 통합 테스트
- **파이프라인 테스트**: 전체 워크플로우 검증
- **성능 테스트**: 처리 속도 및 메모리 사용량 검증
- **호환성 테스트**: 다양한 입력 형식 검증

### 테스트 실행
```bash
# 전체 테스트
python -m pytest tests/ -v

# 성능 테스트
python -m pytest tests/ -m performance -v

# 특정 테스트
python -m pytest tests/test_defect_detection_model.py::DefectDetectionTester::test_model_loading -v
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 불량이 검출되지 않음
- **조명 상태 확인**: 충분한 조명이 있는지 확인
- **카메라 각도 조정**: 제품 표면이 잘 보이는 각도로 조정
- **신뢰도 임계값 낮추기**: `--conf 0.3`으로 임계값 낮추기
- **모델 재학습 고려**: 현재 제품에 맞는 모델 재학습

#### 2. 오탐지가 많음
- **신뢰도 임계값 높이기**: `--conf 0.7`로 임계값 높이기
- **조명 환경 개선**: 일정한 조명 환경 조성
- **모델 재학습**: 더 많은 학습 데이터로 모델 개선

#### 3. 처리 속도가 느림
- **GPU 사용 확인**: GPU가 사용되고 있는지 확인
- **이미지 해상도 낮추기**: 처리 속도 향상을 위해 해상도 조정
- **배치 크기 조정**: 메모리에 맞는 배치 크기 설정

#### 4. 카메라 연결 오류
- **카메라 드라이버 확인**: 최신 드라이버 설치
- **USB 연결 상태 확인**: 안정적인 USB 연결 확인
- **카메라 ID 변경**: 다른 카메라 ID 시도

## 📈 성능 모니터링

### 주요 지표
- **검출 정확도**: Precision, Recall, F1-Score
- **처리 속도**: FPS (Frames Per Second)
- **오탐지율**: False Positive Rate
- **검출 누락율**: False Negative Rate

### 로그 분석
- **로그 파일 위치**: `data/logs/defect_detection.log`
- **검출 결과 통계**: `data/domains/factory/defect_detection/results/`
- **성능 트렌드**: 시간별 성능 변화 모니터링

## 🔒 보안 및 안전

### 산업 보안
- **접근 제어**: 역할 기반 접근 제어 (RBAC)
- **네트워크 보안**: VPN, 방화벽 설정
- **데이터 보호**: 중요 데이터 암호화

### 안전 기능
- **비상 정지**: 비상 상황 시 즉시 정지
- **안전 인터락**: 안전 장치와 연동
- **감시 시스템**: 작업자 안전 모니터링

## 📊 품질 관리 시스템

### 실시간 모니터링
- **불량률 추적**: 실시간 불량률 계산 및 알림
- **품질 지표**: Cp, Cpk, Ppk 등 통계적 품질 지표
- **트렌드 분석**: 시간별 품질 변화 추적

### 알림 시스템
- **불량률 임계값**: 설정된 임계값 초과 시 알림
- **시스템 오류**: 하드웨어/소프트웨어 오류 시 알림
- **성능 저하**: 처리 속도 저하 시 알림

## 🔗 의존성

### 내부 의존성
- `common/`: 공통 유틸리티
- `shared/vision_core/`: 비전 알고리즘 공통 기능
- `config/`: 설정 파일
- `models/weights/`: 모델 가중치

### 외부 의존성
```python
# requirements.txt
opencv-python>=4.5.0
numpy>=1.21.0
onnxruntime>=1.12.0
torch>=1.12.0
torchvision>=0.13.0
pillow>=8.3.0
pyyaml>=6.0
ultralytics>=8.0.0  # YOLO 모델용
```

## 🚀 향후 계획

### 단기 계획 (3개월)
1. **다중 카메라 지원 확장**
2. **불량 검출 성능 최적화**
3. **실시간 대시보드 개발**

### 중기 계획 (6개월)
1. **AI 기반 예측 유지보수**
2. **클라우드 기반 분석**
3. **모바일 앱 연동**

### 장기 계획 (1년)
1. **스마트 팩토리 통합**
2. **자율적 품질 관리 시스템**
3. **글로벌 표준 준수**

## 📞 지원 및 문의

### 문제 해결
1. **기술적 문제**: GitHub Issues 사용
2. **성능 문제**: 성능 모니터링 대시보드 확인
3. **설정 문제**: 설정 파일 문서 참조

### 개발 지원
- **코드 리뷰**: Pull Request 시 자동 리뷰
- **문서화**: 모든 기능에 대한 상세 문서 제공
- **테스트**: 자동화된 테스트 스위트 제공

### 연락처
- **개발팀**: dev@visionsystem.com
- **기술지원**: support@visionsystem.com
- **산업지원**: industry@visionsystem.com

## 📄 라이선스

이 모듈의 코드는 프로젝트 전체 라이선스를 따릅니다.

---

**마지막 업데이트**: 2025-07-04
**버전**: 1.0.0
**작성자**: Vision System Team 