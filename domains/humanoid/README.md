# Humanoid Domain - 인간형 도메인

## 📋 개요

Humanoid 도메인은 인간과 관련된 모든 비전 인식 기능을 담당합니다. 얼굴인식, 감정 검출, 자세 추정 등 인간의 다양한 특성을 분석하고 인식하는 기능들을 포함합니다.

## 🎯 도메인 목적

- **인간 인식**: 얼굴, 감정, 자세 등 인간의 다양한 특성 인식
- **신원 확인**: 얼굴인식을 통한 개인 식별
- **행동 분석**: 자세 추정을 통한 행동 패턴 분석
- **감정 인식**: 표정 분석을 통한 감정 상태 파악

## 🏗️ 도메인 구조

```
humanoid/
├── __init__.py                    # 도메인 패키지 초기화
├── README.md                      # 이 파일
└── face_recognition/              # 얼굴인식 기능
    ├── __init__.py
    ├── README.md                  # 얼굴인식 기능 설명
    ├── run_face_recognition.py    # 실행 스크립트
    ├── models/                    # 모델 클래스들
    │   ├── __init__.py
    │   ├── face_detection_model.py    # 얼굴 검출 모델
    │   └── face_recognition_model.py  # 얼굴 인식 모델
    ├── services/                  # 서비스 클래스들
    │   ├── __init__.py
    │   ├── service.py                 # 기본 서비스
    │   └── face_recognition_service.py # 통합 서비스
    ├── utils/                     # 유틸리티 함수들
    │   ├── __init__.py
    │   └── demo.py                    # 데모 스크립트
    ├── tests/                     # 테스트 파일들
    │   ├── __init__.py
    │   └── test_service.py            # 서비스 테스트
    ├── configs/                   # 설정 파일들
    │   └── __init__.py
    └── pipeline/                  # 9단계 파이프라인
        └── __init__.py
```

## 🚀 포함된 기능들

### 1. Face Recognition (얼굴인식) - 현재 구현됨
- **목적**: 실시간 얼굴 검출 및 인식
- **기술**: ONNX 기반 딥러닝 모델
- **입력**: 웹캠, 이미지 파일, 비디오 파일
- **출력**: 얼굴 위치, 신원 정보, 신뢰도 점수
- **폴더**: `face_recognition/`

### 2. Emotion Detection (감정 검출) - 향후 구현
- **목적**: 얼굴 표정을 통한 감정 상태 분석
- **기술**: CNN 기반 감정 분류 모델
- **입력**: 얼굴 이미지
- **출력**: 감정 카테고리, 신뢰도 점수
- **폴더**: `emotion_detection/` (향후 생성)

### 3. Pose Estimation (자세 추정) - 향후 구현
- **목적**: 인체 자세 및 관절 위치 추정
- **기술**: Keypoint Detection 모델
- **입력**: 인체 이미지
- **출력**: 관절 좌표, 자세 정보
- **폴더**: `pose_estimation/` (향후 생성)

## 📊 데이터 관리

### 학습 데이터
```
datasets/humanoid/
├── face_recognition/              # 얼굴인식 학습 데이터
│   ├── raw/                       # 원본 이미지
│   ├── processed/                 # 전처리된 이미지
│   ├── annotations/               # 라벨링 데이터
│   └── splits/                    # train/val/test 분할
├── emotion_detection/             # 감정 검출 학습 데이터 (향후)
└── pose_estimation/               # 자세 추정 학습 데이터 (향후)
```

### 런타임 데이터
```
data/domains/humanoid/
├── face_recognition/              # 얼굴인식 런타임 데이터
│   ├── captures/                  # 캡처된 이미지
│   ├── results/                   # 처리 결과
│   └── logs/                      # 실행 로그
├── emotion_detection/             # 감정 검출 런타임 데이터 (향후)
└── pose_estimation/               # 자세 추정 런타임 데이터 (향후)
```

## 🔧 기술 스택

### 핵심 기술
- **딥러닝 프레임워크**: PyTorch (학습), ONNX Runtime (추론)
- **컴퓨터 비전**: OpenCV
- **이미지 처리**: NumPy, PIL
- **데이터 관리**: JSON, YAML

### 모델 아키텍처
- **얼굴 검출**: RetinaFace, MTCNN
- **얼굴 인식**: ArcFace, FaceNet
- **감정 검출**: ResNet, EfficientNet (향후)
- **자세 추정**: HRNet, PoseNet (향후)

## 🚀 빠른 시작

### 얼굴인식 실행
```bash
# 기본 실행 (웹캠 사용)
python domains/humanoid/face_recognition/run_face_recognition.py

# 이미지 파일로 테스트
python domains/humanoid/face_recognition/run_face_recognition.py --source path/to/image.jpg

# 결과 화면 표시
python domains/humanoid/face_recognition/run_face_recognition.py --show

# 결과 저장
python domains/humanoid/face_recognition/run_face_recognition.py --save
```

### 테스트 실행
```bash
# 얼굴인식 테스트
python -m pytest domains/humanoid/face_recognition/tests/ -v

# 특정 테스트
python -m pytest domains/humanoid/face_recognition/tests/test_service.py::FaceRecognitionTester::test_face_detection_model -v
```

## 📈 성능 지표

### 얼굴인식 성능
- **검출 정확도**: 95% 이상
- **인식 정확도**: 90% 이상
- **처리 속도**: 30 FPS 이상 (GPU)
- **메모리 사용량**: 2GB 이하

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
python -m pytest domains/humanoid/ -v

# 성능 테스트
python -m pytest domains/humanoid/ -m performance -v

# 통합 테스트
python -m pytest tests/test_humanoid_integration.py -v
```

## 🔧 개발 가이드

### 새로운 기능 추가
1. **기능 폴더 생성**: `domains/humanoid/new_feature/`
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
# config/humanoid.yaml
face_recognition:
  detection:
    confidence_threshold: 0.5
    min_face_size: 80
  recognition:
    confidence_threshold: 0.6
    embedding_dim: 512
  performance:
    target_fps: 30
    max_batch_size: 4
```

### 환경별 설정
- **개발 환경**: `config/environments/development_humanoid.yaml`
- **테스트 환경**: `config/environments/test_humanoid.yaml`
- **운영 환경**: `config/environments/production_humanoid.yaml`

## 🔒 보안 및 개인정보 보호

### GDPR 준수
- **데이터 익명화**: 기본적으로 얼굴 데이터 익명화
- **동의 관리**: 사용자 동의 기반 데이터 처리
- **보존 정책**: 30일 자동 삭제 정책
- **암호화**: 모든 개인정보 암호화 저장

### 보안 기능
- **접근 제어**: 역할 기반 접근 제어 (RBAC)
- **감사 로그**: 모든 데이터 접근 로그 기록
- **암호화**: 전송 및 저장 데이터 암호화

## 📊 모니터링 및 로깅

### 성능 모니터링
- **실시간 메트릭**: FPS, 정확도, 지연시간
- **리소스 모니터링**: CPU, GPU, 메모리 사용량
- **알림 시스템**: 성능 저하 시 자동 알림

### 로깅 시스템
- **구조화 로깅**: JSON 형식 로그
- **로그 레벨**: DEBUG, INFO, WARNING, ERROR
- **로그 보존**: 30일 자동 보존

## 🔗 관련 문서

- [프로젝트 전체 README](../../README.md)
- [도메인 구조 가이드](../README.md)
- [얼굴인식 기능 문서](./face_recognition/README.md)
- [공통 모듈 문서](../../common/README.md)
- [공유 모듈 문서](../../shared/README.md)

## 🚀 향후 계획

### 단기 계획 (3개월)
1. **감정 검출 기능 구현**
2. **얼굴인식 성능 최적화**
3. **모바일 지원 추가**

### 중기 계획 (6개월)
1. **자세 추정 기능 구현**
2. **멀티 얼굴 인식 지원**
3. **실시간 스트리밍 지원**

### 장기 계획 (1년)
1. **3D 얼굴 모델링**
2. **행동 분석 기능**
3. **AI 기반 개인화 서비스**

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
- **보안문의**: security@visionsystem.com 