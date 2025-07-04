# Factory Domain - 공장 도메인

## 📋 개요

Factory 도메인은 제조업 관련 모든 비전 검사 기능을 담당합니다. 불량 검출, 품질 평가, 재고 추적 등 제조 과정에서 발생하는 다양한 비전 인식 요구사항을 해결합니다.

## 🎯 도메인 목적

- **품질 관리**: 제품 불량 자동 검출 및 분류
- **생산성 향상**: 실시간 품질 모니터링을 통한 생산 효율성 증대
- **비용 절감**: 수동 검사 대비 검사 비용 절감
- **일관성 보장**: 24시간 일관된 품질 검사 수행

## 🏗️ 도메인 구조

```
factory/
├── __init__.py                    # 도메인 패키지 초기화
├── README.md                      # 이 파일
└── defect_detection/              # 불량 검출 기능
    ├── __init__.py
    ├── README.md                  # 불량 검출 기능 설명
    ├── run_defect_detection.py    # 실행 스크립트
    ├── models/                    # 모델 클래스들
    │   ├── __init__.py
    │   └── defect_detection_model.py  # 불량 검출 모델
    ├── services/                  # 서비스 클래스들
    │   └── __init__.py
    ├── utils/                     # 유틸리티 함수들
    │   └── __init__.py
    ├── tests/                     # 테스트 파일들
    │   ├── __init__.py
    │   └── test_defect_detection_model.py  # 모델 테스트
    ├── configs/                   # 설정 파일들
    │   └── __init__.py
    └── pipeline/                  # 9단계 파이프라인
        └── __init__.py
```

## 🚀 포함된 기능들

### 1. Defect Detection (불량 검출) - 현재 구현됨
- **목적**: 제품 불량 자동 검출 및 분류
- **기술**: YOLO 기반 객체 검출 모델
- **입력**: 제품 이미지, 카메라 스트림
- **출력**: 불량 위치, 불량 유형, 신뢰도 점수
- **폴더**: `defect_detection/`

### 2. Quality Assessment (품질 평가) - 향후 구현
- **목적**: 제품 품질 수준 자동 평가
- **기술**: CNN 기반 품질 분류 모델
- **입력**: 제품 이미지
- **출력**: 품질 등급, 품질 점수, 개선 제안
- **폴더**: `quality_assessment/` (향후 생성)

### 3. Inventory Tracking (재고 추적) - 향후 구현
- **목적**: 제품 재고 자동 추적 및 관리
- **기술**: 객체 추적 및 카운팅 모델
- **입력**: 창고/생산라인 영상
- **출력**: 제품 수량, 위치 정보, 이동 경로
- **폴더**: `inventory_tracking/` (향후 생성)

## 📊 데이터 관리

### 학습 데이터
```
datasets/factory/
├── defect_detection/              # 불량 검출 학습 데이터
│   ├── raw/                       # 원본 이미지
│   ├── processed/                 # 전처리된 이미지
│   ├── annotations/               # 라벨링 데이터 (YOLO 형식)
│   └── splits/                    # train/val/test 분할
├── quality_assessment/            # 품질 평가 학습 데이터 (향후)
└── inventory_tracking/            # 재고 추적 학습 데이터 (향후)
```

### 런타임 데이터
```
data/domains/factory/
├── defect_detection/              # 불량 검출 런타임 데이터
│   ├── captures/                  # 캡처된 이미지
│   ├── results/                   # 검출 결과
│   └── logs/                      # 실행 로그
├── quality_assessment/            # 품질 평가 런타임 데이터 (향후)
└── inventory_tracking/            # 재고 추적 런타임 데이터 (향후)
```

## 🔧 기술 스택

### 핵심 기술
- **딥러닝 프레임워크**: PyTorch (학습), ONNX Runtime (추론)
- **객체 검출**: YOLO, Faster R-CNN
- **컴퓨터 비전**: OpenCV
- **이미지 처리**: NumPy, PIL
- **데이터 관리**: JSON, YAML, SQLite

### 모델 아키텍처
- **불량 검출**: YOLOv8, YOLOv5, EfficientDet
- **품질 평가**: ResNet, EfficientNet (향후)
- **재고 추적**: DeepSORT, ByteTrack (향후)

## 🚀 빠른 시작

### 불량 검출 실행
```bash
# 기본 실행 (웹캠 사용)
python domains/factory/defect_detection/run_defect_detection.py

# 이미지 파일로 테스트
python domains/factory/defect_detection/run_defect_detection.py --source path/to/image.jpg

# 결과 화면 표시
python domains/factory/defect_detection/run_defect_detection.py --show

# 결과 저장
python domains/factory/defect_detection/run_defect_detection.py --save

# 설정 파일 지정
python domains/factory/defect_detection/run_defect_detection.py --config config/defect_detection.yaml
```

### 테스트 실행
```bash
# 불량 검출 테스트
python -m pytest domains/factory/defect_detection/tests/ -v

# 특정 테스트
python -m pytest domains/factory/defect_detection/tests/test_defect_detection_model.py -v
```

## 📈 성능 지표

### 불량 검출 성능
- **검출 정확도**: 95% 이상
- **분류 정확도**: 90% 이상
- **처리 속도**: 25 FPS 이상 (GPU)
- **메모리 사용량**: 3GB 이하

### 하드웨어 요구사항
- **최소 사양**: CPU 4코어, RAM 8GB
- **권장 사양**: GPU 4GB+, RAM 16GB
- **최적 사양**: GPU 8GB+, RAM 32GB

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

## 🧪 테스트 전략

### 단위 테스트
- **모델 테스트**: 각 모델의 정확도 및 성능 검증
- **서비스 테스트**: 비즈니스 로직 검증
- **유틸리티 테스트**: 헬퍼 함수 검증

### 통합 테스트
- **파이프라인 테스트**: 전체 워크플로우 검증
- **성능 테스트**: 처리 속도 및 메모리 사용량 검증
- **호환성 테스트**: 다양한 입력 형식 검증

### 테스트 실행
```bash
# 전체 테스트
python -m pytest domains/factory/ -v

# 성능 테스트
python -m pytest domains/factory/ -m performance -v

# 통합 테스트
python -m pytest tests/test_factory_integration.py -v
```

## 🔧 개발 가이드

### 새로운 기능 추가
1. **기능 폴더 생성**: `domains/factory/new_feature/`
2. **표준 구조 생성**: models/, services/, utils/, tests/, configs/, pipeline/
3. **모델 구현**: ONNX 기반 모델 클래스
4. **서비스 구현**: 비즈니스 로직 서비스
5. **테스트 작성**: 단위 테스트 및 통합 테스트
6. **문서 작성**: README.md 및 API 문서

### 코드 품질 관리
- **Type Hints**: 모든 함수에 타입 힌트 필수
- **Docstring**: Google Style 문서화 필수
- **테스트 커버리지**: 80% 이상 유지
- **코드 리뷰**: 모든 변경사항에 대한 리뷰 필수

## 📝 설정 관리

### 도메인별 설정
```yaml
# config/factory.yaml
defect_detection:
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
```

### 환경별 설정
- **개발 환경**: `config/environments/development_factory.yaml`
- **테스트 환경**: `config/environments/test_factory.yaml`
- **운영 환경**: `config/environments/production_factory.yaml`

## 🏭 산업 표준 준수

### 품질 관리 표준
- **ISO 9001**: 품질 관리 시스템
- **ISO 14001**: 환경 관리 시스템
- **IATF 16949**: 자동차 산업 품질 관리

### 안전 표준
- **ISO 13849**: 기계 안전
- **IEC 61508**: 기능 안전
- **CE 마킹**: 유럽 안전 표준

## 📊 품질 관리 시스템

### 실시간 모니터링
- **불량률 추적**: 실시간 불량률 계산 및 알림
- **품질 지표**: Cp, Cpk, Ppk 등 통계적 품질 지표
- **트렌드 분석**: 시간별 품질 변화 추적

### 알림 시스템
- **불량률 임계값**: 설정된 임계값 초과 시 알림
- **시스템 오류**: 하드웨어/소프트웨어 오류 시 알림
- **성능 저하**: 처리 속도 저하 시 알림

## 🔒 보안 및 안전

### 산업 보안
- **접근 제어**: 역할 기반 접근 제어 (RBAC)
- **네트워크 보안**: VPN, 방화벽 설정
- **데이터 보호**: 중요 데이터 암호화

### 안전 기능
- **비상 정지**: 비상 상황 시 즉시 정지
- **안전 인터락**: 안전 장치와 연동
- **감시 시스템**: 작업자 안전 모니터링

## 📊 모니터링 및 로깅

### 성능 모니터링
- **실시간 메트릭**: FPS, 정확도, 지연시간
- **리소스 모니터링**: CPU, GPU, 메모리 사용량
- **알림 시스템**: 성능 저하 시 자동 알림

### 로깅 시스템
- **구조화 로깅**: JSON 형식 로그
- **로그 레벨**: DEBUG, INFO, WARNING, ERROR
- **로그 보존**: 90일 자동 보존 (산업 표준)

## 🔗 관련 문서

- [프로젝트 전체 README](../../README.md)
- [도메인 구조 가이드](../README.md)
- [불량 검출 기능 문서](./defect_detection/README.md)
- [공통 모듈 문서](../../common/README.md)
- [공유 모듈 문서](../../shared/README.md)

## 🚀 향후 계획

### 단기 계획 (3개월)
1. **품질 평가 기능 구현**
2. **불량 검출 성능 최적화**
3. **다중 카메라 지원**

### 중기 계획 (6개월)
1. **재고 추적 기능 구현**
2. **실시간 대시보드 개발**
3. **AI 기반 예측 유지보수**

### 장기 계획 (1년)
1. **스마트 팩토리 통합**
2. **클라우드 기반 분석**
3. **자율적 품질 관리 시스템**

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