# Vision System

비전 시스템 기반 얼굴인식, 공장 불량 검출, 활선 검사 등을 위한 크로스 플랫폼 호환 시스템입니다.

## 🏗️ 프로젝트 구조

```
vision_system/
├── domains/                   # 도메인별 독립 개발
│   ├── humanoid/              # 인간형 로봇 관련
│   │   └── face_recognition/  # 얼굴인식 도메인
│   ├── factory/               # 공장 자동화 관련
│   │   └── defect_detection/  # 불량 검출 도메인 (향후)
│   └── infrastructure/        # 인프라 관련
│       └── powerline_inspection/ # 활선 검사 도메인 (향후)
├── shared/                    # 공통 모듈
│   ├── vision_core/           # 비전 알고리즘 공통 모듈
│   └── security/              # 보안 모듈
├── common/                    # 범용 유틸리티 및 기능
├── config/                    # 전역 설정 관리
├── models/                    # 모델 저장소
│   ├── weights/               # 모델 가중치
│   ├── metadata/              # 모델 메타데이터
│   └── configs/               # 모델 설정
├── datasets/                  # 🎯 학습 전용 데이터 (ML 데이터셋)
├── data/                      # 🎯 런타임 전용 데이터
│   ├── temp/                  # 임시 파일
│   ├── logs/                  # 로그 파일
│   └── output/                # 결과물 저장
├── docs/                      # 문서
├── scripts/                   # 유틸리티 스크립트
├── tests/                     # 테스트 코드
├── requirements.txt           # 의존성 정의
├── pyproject.toml            # 프로젝트 설정
└── README.md                 # 프로젝트 개요
```

## 🎯 지원 도메인

### 현재 개발 중
- **humanoid/face_recognition**: 얼굴인식 시스템

### 향후 개발 예정
- **factory/defect_detection**: 공장 불량 검출 (YOLOv8 기반)
- **infrastructure/powerline_inspection**: 활선 상태 검사

## 🔧 기술 스택

- **Python**: 3.10+ (3.13 호환)
- **AI Framework**: PyTorch (학습), ONNX (추론)
- **Computer Vision**: OpenCV
- **Platform**: Windows, Ubuntu, NVIDIA Jetson
- **Hardware**: CPU, CUDA GPU, TensorRT

## 🚀 시작하기

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 하드웨어 검증
```bash
python scripts/validation/validate_hardware_connection.py
```

### 3. 얼굴인식 실행
```bash
python domains/humanoid/face_recognition/run.py
```

## 📋 개발 규칙

- 모든 경로는 `pathlib.Path` 사용 (크로스 플랫폼 호환)
- 하드웨어 환경 자동 감지 및 최적화
- Python 3.10~3.13+ 호환성 보장
- ONNX 모델 사용 (Jetson 호환)

## 📖 문서

자세한 내용은 [docs/](docs/) 폴더를 참조하세요. 