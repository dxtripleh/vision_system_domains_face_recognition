# Vision System - Face Recognition

비전 시스템의 얼굴인식 도메인을 위한 고성능 AI 시스템입니다.

## 🎯 새로운 모델 아키텍처

### 모델 선택 전략
- **1차 선택**: RetinaFace MobileNet0.25 → ONNX → ONNX Runtime
- **2차 선택**: MobileFaceNet → ONNX → ONNX Runtime  
- **백업**: OpenCV Haar Cascade + MobileFaceNet

### 성능 최적화
- ONNX Runtime을 통한 하드웨어 가속
- 자동 디바이스 감지 (CPU/GPU)
- 폴백 전략으로 안정성 보장

## 📁 프로젝트 구조

```
vision_system/
├── domains/                   # 도메인별 독립 개발
│   └── face_recognition/      # 얼굴인식 도메인
│       ├── core/              # 도메인 핵심 로직 (DDD)
│       ├── infrastructure/    # 인프라 계층
│       └── interfaces/        # API 인터페이스
├── shared/                    # 공통 모듈
│   ├── vision_core/           # 비전 알고리즘 공통 모듈
│   └── security/              # 보안 모듈
├── common/                    # 범용 유틸리티
├── config/                    # 전역 설정 관리
├── models/                    # 모델 저장소
│   ├── weights/               # 모델 가중치
│   ├── metadata/              # 모델 메타데이터
│   └── configs/               # 모델 설정
├── datasets/                  # 학습 전용 데이터
├── data/                      # 런타임 전용 데이터
├── scripts/                   # 유틸리티 스크립트
└── tests/                     # 테스트 코드
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements/production.txt
```

### 2. 모델 다운로드
```bash
# 모든 모델 다운로드
python scripts/development/model_management/download_models.py

# 1차 선택 모델만 다운로드
python scripts/development/model_management/download_models.py --primary-only
```

### 3. 시스템 실행
```bash
# API 서버 실행
python -m domains.face_recognition.interfaces.api.face_recognition_api

# CLI 인터페이스
python -m domains.face_recognition.interfaces.cli.face_recognition_cli
```

## 🔧 설정

### 모델 설정
`config/face_recognition_api.yaml`에서 모델 경로와 파라미터를 설정할 수 있습니다:

```yaml
models:
  detection:
    primary:
      name: "retinaface_mobilenet025"
      path: "models/weights/face_detection_retinaface_mobilenet025_20250703.onnx"
      device: "auto"
    backup:
      name: "opencv_haarcascade"
      path: "models/weights/face_detection_opencv_haarcascade_20250628.xml"
```

### 성능 설정
```yaml
performance:
  onnxruntime:
    execution_mode: "auto"
    graph_optimization_level: "all"
  gpu:
    provider: "cuda"
    memory_fraction: 0.8
```

## 📊 성능 벤치마크

### 모델별 성능 (예상)
| 모델 | FPS (CPU) | FPS (GPU) | 정확도 | 메모리 사용량 |
|------|-----------|-----------|--------|---------------|
| RetinaFace MobileNet0.25 | 15 | 45 | 95% | 50MB |
| MobileFaceNet | 20 | 60 | 98% | 30MB |
| OpenCV Haar Cascade | 8 | 8 | 85% | 10MB |

## 🔒 보안 기능

- **데이터 보호**: GDPR 준수, 얼굴 데이터 익명화
- **암호화**: 모델 파일 암호화, API 통신 암호화
- **인증**: JWT 토큰 기반 인증
- **감사**: 보안 이벤트 로깅

## 📈 모니터링

### Prometheus 메트릭
- 프레임 처리율
- 검출 정확도
- 시스템 리소스 사용량
- 에러율

### Grafana 대시보드
- 실시간 성능 모니터링
- 이상 감지 알림
- 비즈니스 메트릭 추적

## 🧪 테스트

```bash
# 단위 테스트
pytest tests/unit/

# 통합 테스트
pytest tests/integration/

# 성능 테스트
pytest tests/performance/
```

## 📝 개발 가이드

### 새로운 모델 추가
1. `models/weights/`에 모델 파일 추가
2. `config/face_recognition_api.yaml`에 설정 추가
3. `models/metadata/models_metadata.json`에 메타데이터 추가
4. 해당 모델 클래스 구현

### 코드 스타일
```bash
# 코드 포맷팅
black .
isort .

# 린팅
flake8 .
mypy .
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🆘 지원

- **문서**: [docs/](docs/) 폴더 참조
- **이슈**: GitHub Issues 사용
- **토론**: GitHub Discussions 사용

## 🔄 변경 이력

### v2.0.0 (2025-07-03)
- 새로운 ONNX Runtime 기반 모델 아키텍처 도입
- RetinaFace MobileNet0.25 1차 선택 모델로 변경
- MobileFaceNet 2차 선택 모델로 변경
- OpenCV Haar Cascade 백업 모델로 유지
- 프로젝트 구조 정리 및 archive 폴더로 기존 파일 이동

### v1.0.0 (2025-06-28)
- 초기 얼굴인식 시스템 구현
- OpenCV 기반 검출 및 인식
- 기본 API 및 CLI 인터페이스 제공